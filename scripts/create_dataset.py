import openai
from openai import AsyncOpenAI
from pydantic import BaseModel
from dotenv import load_dotenv
import json
import asyncio
import random
import yaml

load_dotenv(override=True)

with open("config.yml") as f:
    config = yaml.safe_load(f)

gen_config = config["data"]["generation"]
LLM_MODEL = gen_config["llm_model"]
CONCURRENT = gen_config["concurrent"]
NUM_TOPICS = gen_config["num_topics"]
NUM_SAMPLES = gen_config["num_samples"]
LABELS_PER_SAMPLE = gen_config["labels_per_sample"]
BATCH_SIZE = gen_config["batch_size"]
OUTPUT_PATH = config["data"]["path"]


# --- Pydantic models for structured output ---

class TopicList(BaseModel):
    topics: list[str]

class Sample(BaseModel):
    text: str
    labels: list[str]

class SampleList(BaseModel):
    samples: list[Sample]


# --- Prompts ---

TOPIC_PROMPT = """
Generate broad, high-level topics. Each should be a single wide category (1-2 words).
They must be diverse, spanning completely different areas of knowledge.
Each must be unique. Do not repeat any previously generated topics.

Examples: Animals, Finance, Education, Sports, Technology, Medicine, Politics, Music
"""

SAMPLE_PROMPT = """
You are generating training data for a zero-shot text classification model.

For each group of labels below, write one simple, short sentence that relates to ALL the given labels.
Keep sentences plain and direct. Use only basic ASCII characters — no special dashes, quotes, or unicode symbols.
Write at a middle-school reading level. One clear idea per sentence.

Example:
Labels: ["Finance", "Technology", "Education"]
-> text: "Many coding schools let students pay after they get a job in tech." labels: ["Finance", "Technology", "Education"]

Labels: ["Animals", "Environment"]
-> text: "Polar bears are losing their habitat as arctic ice melts." labels: ["Animals", "Environment"]
"""


async def get_topics(client: AsyncOpenAI) -> list[str]:
    topics = set()
    messages = [
        {"role": "system", "content": TOPIC_PROMPT},
        {"role": "user", "content": f"Generate {NUM_TOPICS} unique topics."}
    ]

    while len(topics) < NUM_TOPICS:
        response = await client.beta.chat.completions.parse(
            model=LLM_MODEL,
            messages=messages,
            response_format=TopicList
        )

        assistant_msg = response.choices[0].message
        new = set(assistant_msg.parsed.topics) - topics
        topics.update(new)
        print(f"Topics so far: {len(topics)}/{NUM_TOPICS}")

        if len(topics) < NUM_TOPICS:
            remaining = NUM_TOPICS - len(topics)
            messages.append({"role": "assistant", "content": assistant_msg.content})
            messages.append({"role": "user", "content": f"I still need {remaining} more unique topics."})

    return list(topics)[:NUM_TOPICS]


def make_label_combos(topics: list[str], num_samples: int, labels_per_sample: int) -> list[list[str]]:
    """Pick random combinations of topics as label sets."""
    return [sorted(random.sample(topics, labels_per_sample)) for _ in range(num_samples)]


async def generate_batch(client: AsyncOpenAI, label_groups: list[list[str]], semaphore: asyncio.Semaphore) -> list[dict]:
    async with semaphore:
        groups_text = "\n".join(
            f'{i+1}. Labels: {json.dumps(labels)}'
            for i, labels in enumerate(label_groups)
        )
        messages = [
            {"role": "system", "content": SAMPLE_PROMPT},
            {"role": "user", "content": f"Generate one sentence for each group:\n{groups_text}"}
        ]

        response = await client.beta.chat.completions.parse(
            model=LLM_MODEL,
            messages=messages,
            response_format=SampleList
        )

        samples = response.choices[0].message.parsed.samples
        # use the original labels, not whatever the model returned
        # strip unicode to keep text ASCII-clean for BERT tokenizer
        results = []
        for sample, original_labels in zip(samples, label_groups):
            clean_text = sample.text.encode('ascii', 'replace').decode('ascii')
            results.append({"text": clean_text, "labels": original_labels})
        return results


async def main():
    client = AsyncOpenAI()
    try:
        topics = await get_topics(client)
        print(f"Got {len(topics)} topics: {topics}")

        combos = make_label_combos(topics, NUM_SAMPLES, LABELS_PER_SAMPLE)
        print(f"Generated {len(combos)} unique label combinations")

        # split into batches and generate concurrently
        semaphore = asyncio.Semaphore(CONCURRENT)
        batches = [combos[i:i + BATCH_SIZE] for i in range(0, len(combos), BATCH_SIZE)]
        tasks = [generate_batch(client, batch, semaphore) for batch in batches]
        results = await asyncio.gather(*tasks)

        final_data = [sample for batch in results for sample in batch]
        print(f"Generated {len(final_data)} samples")

        with open(OUTPUT_PATH, 'w') as f:
            json.dump(final_data, f, indent=2)

        print(f"Done! {len(final_data)} samples saved to {OUTPUT_PATH}")

    except openai.APIConnectionError:
        print("Error: could not reach the OpenAI API")
    except openai.RateLimitError:
        print("Error: rate limit hit")


if __name__ == "__main__":
    asyncio.run(main())
