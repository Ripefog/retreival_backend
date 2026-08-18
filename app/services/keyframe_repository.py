from pathlib import Path


class KeyframeRepository:

    def __init__(self, root_dir: str = "keyframes"):
        self.root_dir = Path(root_dir)

    def get_image_path(self, keyframe_id: str) -> Path:
        """
        Convert keyframe_id to local image path.

        Example:
            L21_V001_0000.00s.jpg

        ->  keyframes/L21/L21_V001/L21_V001_0000.00s.jpg
        """

        filename = Path(keyframe_id).name

        # L21_V001_0000.00s.jpg
        #        ↓ split
        parts = filename.split("_")

        if len(parts) < 3:
            raise ValueError(
                f"Invalid keyframe_id: {keyframe_id}"
            )

        dataset_id = parts[0]       # L21
        video_id = f"{parts[0]}_{parts[1]}"  # L21_V001

        image_path = (
            self.root_dir
            / dataset_id
            / video_id
            / filename
        )

        return image_path


keyframe_repository = KeyframeRepository()

# def main():
#     print("Testing KeyframeRepository...")
#     repo = KeyframeRepository()
#     keyframe_id = "L22_V001_0000.00s.jpg"


#     image_path = repo.get_image_path(keyframe_id)
#     print(image_path)

# if __name__ == "__main__":
#     main()