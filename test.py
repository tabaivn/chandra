"""
Script inference OCR: xử lý một thư mục ảnh (theo batch) hoặc một ảnh đơn.
Dựa trên tests/integration/test_image_inference.py và tests/conftest.py.
"""
import argparse
from pathlib import Path
from typing import List

from chandra.input import load_image
from chandra.model import InferenceManager
from chandra.model.schema import BatchInputItem


# Các định dạng ảnh hỗ trợ (chỉ ảnh, không PDF)
IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".tiff", ".bmp",
}


def get_image_files(input_path: Path) -> List[Path]:
    """Lấy danh sách file ảnh từ đường dẫn (file hoặc thư mục)."""
    if input_path.is_file():
        if input_path.suffix.lower() in IMAGE_EXTENSIONS:
            return [input_path]
        raise ValueError(f"Định dạng không hỗ trợ: {input_path.suffix}")
    if input_path.is_dir():
        files = []
        for ext in IMAGE_EXTENSIONS:
            files.extend(input_path.glob(f"*{ext}"))
            files.extend(input_path.glob(f"*{ext.upper()}"))
        return sorted(files)
    raise ValueError(f"Đường dẫn không tồn tại: {input_path}")


def run_ocr_single_image(
    image_path: Path,
    manager: InferenceManager,
    max_output_tokens: int = 128,
) -> str:
    """OCR một ảnh, trả về markdown."""
    img = load_image(str(image_path))
    batch = [BatchInputItem(image=img, prompt_type="ocr_layout")]
    outputs = manager.generate(batch, max_output_tokens=max_output_tokens)
    return outputs[0].markdown


def run_ocr_directory(
    input_dir: Path,
    manager: InferenceManager,
    batch_size: int = 16,
    max_output_tokens: int = 128,
    output_dir: Path | None = None,
) -> List[tuple[Path, str]]:
    """
    OCR toàn bộ ảnh trong thư mục theo batch.
    Trả về list (path_ảnh, markdown).
    Nếu output_dir được chỉ định, ghi mỗi kết quả ra file <tên_ảnh>.md.
    """
    image_paths = get_image_files(input_dir)
    if not image_paths:
        print("Không tìm thấy ảnh nào.")
        return []

    results: List[tuple[Path, str]] = []
    out_dir = output_dir or input_dir
    import time
    start = time.time()
    for batch_start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[batch_start : batch_start + batch_size]
        images = [load_image(str(p)) for p in batch_paths]
        batch = [
            BatchInputItem(image=img, prompt_type="ocr_layout")
            for img in images
        ]
        outputs = manager.generate(batch, max_output_tokens=max_output_tokens)

        for path, out in zip(batch_paths, outputs):
            results.append((path, out.markdown))
            if output_dir is not None or out_dir != input_dir:
                out_path = out_dir / f"{path.stem}.md"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(out.markdown, encoding="utf-8")

    return results, time.time() - start


def main():
    parser = argparse.ArgumentParser(
        description="OCR ảnh: một file hoặc cả thư mục (batch 16 ảnh)."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Đường dẫn tới một ảnh hoặc thư mục chứa ảnh",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Số ảnh mỗi batch khi xử lý thư mục (mặc định: 16)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=None,
        help="Thư mục ghi kết quả .md (khi input là thư mục). Mặc định: cùng thư mục với input",
    )
    parser.add_argument(
        "--method",
        choices=("hf", "vllm"),
        default="hf",
        help="Phương thức inference (mặc định: hf)",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=128,
        help="Số token tối đa mỗi ảnh (mặc định: 128)",
    )
    args = parser.parse_args()

    input_path = args.input.resolve()
    if not input_path.exists():
        raise SystemExit(f"Không tìm thấy: {input_path}")

    print(f"Đang load model (method={args.method})...")
    manager = InferenceManager(method=args.method)
    print("Model đã sẵn sàng.")

    if input_path.is_file():
        print(f"OCR ảnh: {input_path}")
        markdown = run_ocr_single_image(
            input_path, manager, max_output_tokens=args.max_output_tokens
        )
        print(markdown)
        return

    # Thư mục
    print(f"OCR thư mục: {input_path} (batch_size={args.batch_size})")
    results, time_inference = run_ocr_directory(
        input_path,
        manager,
        batch_size=args.batch_size,
        max_output_tokens=args.max_output_tokens,
        output_dir=args.output_dir,
    )
    print(f"Đã xử lý {len(results)} ảnh.")
    for path, md in results:
        print(f"  - {path.name}")

    print("Thời gian xử lý: ", time_inference)


if __name__ == "__main__":
    main()
