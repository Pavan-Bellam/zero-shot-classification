from model import load_model
from pathlib import Path
from huggingface_hub import HfApi
import argparse
import tempfile



def main():
    parser = argparse.ArgumentParser(description="Push a trained model to HuggingFace Hub")
    parser.add_argument("--run", type=str, required=True, help="Path to a saved run directory")
    parser.add_argument("--repo_id", type=str, required=True, help="HuggingFace repo id (e.g. username/model-name)")
    parser.add_argument("--public", action="store_true", help="Make the repo public (default is private)")
    args = parser.parse_args()

    run_path = Path(args.run)
    if not run_path.exists():
        print(f"Error: {run_path} does not exist")
        return

    private = not args.public

    print(f"Loading from {run_path}...")
    model = load_model(run_path)

    print(f"Pushing to {args.repo_id} (private={private})...")
    api = HfApi()
    api.create_repo(args.repo_id, private=private, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        model.save_pretrained(tmp)
        api.upload_folder(folder_path=tmp, repo_id=args.repo_id)
    print(f"Done! Model available at https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
