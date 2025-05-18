import os
import cv2
import sys
from pathlib import Path
from utils import Binarization, LineSegmentation, Scanner, WordSegmentation

def main(src_path, out_path):
    os.makedirs(out_path, exist_ok=True)
    words_path = Path(out_path) / "words"
    words_path.mkdir(parents=True, exist_ok=True)

    name = Path(out_path).stem
    extension = ".png"

    # Step 1: Crop
    image = cv2.imread(src_path)
    scanner = Scanner()
    image_cropped = scanner.process(image)

    save_crop = Path(out_path) / f"{name}_1_crop{extension}"
    cv2.imwrite(str(save_crop), image_cropped)

    # Step 1.1: Resize
    new_w = 1280
    new_h = int((new_w * image_cropped.shape[0]) / image_cropped.shape[1])
    image_cropped = cv2.resize(image_cropped, (new_w, new_h))

    chunks_number = 8
    chunks_process = 4

    # Step 2: Binarization
    threshold = Binarization()
    image_binary = threshold.binarize(image_cropped, method=3)

    save_binary = Path(out_path) / f"{name}_2_binary{extension}"
    cv2.imwrite(str(save_binary), image_binary)

    # Step 3: Line Segmentation
    line_segmenter = LineSegmentation()
    image_lines = image_binary.copy()
    lines = line_segmenter.segment(image_lines, chunks_number, chunks_process)

    save_lines = Path(out_path) / f"{name}_3_lines{extension}"
    cv2.imwrite(str(save_lines), image_lines)

    # Step 4: Word Segmentation
    word_segmenter = WordSegmentation()
    word_segmenter.set_kernel(11, 11, 7)
    summary = []

    for i, line_img in enumerate(lines):
        line_index = f"{(i+1)*1e-6:.6f}"[5:]

        words = word_segmenter.segment(line_img)
        if not words:
            continue

        summary.append(words[0])
        for j, word_img in enumerate(words[1:]):
            word_index = f"{line_index}_{(j+1)*1e-6:.6f}"[5:]
            save_word = words_path / f"{word_index}{extension}"
            cv2.imwrite(str(save_word), word_img)

    for i, summary_img in enumerate(summary):
        summary_index = f"_4_summary_{(i+1)*1e-6:.6f}"[5:]
        save_summary = Path(out_path) / f"{name}{summary_index}{extension}"
        cv2.imwrite(str(save_summary), summary_img)


if __name__ == "__main__":
    # src_path = sys.argv[1]
    # out_path = sys.argv[2]
    src_path = "../sample_docs/001.png"
    out_path = "results/1_png"
    main(src_path, out_path)
