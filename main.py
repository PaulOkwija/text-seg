# main.py
import os
import cv2
from segment import sauvola_binarization
from seg_logic import WordSegmentation
from line_seg_logic import line_segmentation, get_horizontal_projection, find_valleys

INPUT_FOLDER = 'sample_docs'
OUTPUT_words = 'results/words'
OUTPUT_lines = 'results/lines'
OUTPUT_FOLDER = 'results'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

for img_file in os.listdir(INPUT_FOLDER):
    if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    print(f"Processing {img_file}...")
    img_path = os.path.join(INPUT_FOLDER, img_file)
    bin_img = sauvola_binarization(img_path)
    lines = line_segmentation(bin_img)
    print(f"Found {len(lines)} lines in {img_file}.")
    for i, line in enumerate(lines):
        line_path = os.path.join(OUTPUT_lines, f"{img_file}_line_{i}.png")
        cv2.imwrite(line_path, line)

        ws = WordSegmentation()
        ws.set_kernel(11, 11, 7)
        words = ws.segment(line)
        
        # words = segment_words(line)
        for j, word in enumerate(words):
            word_path = os.path.join(OUTPUT_words, f"{img_file}_line_{i}_word_{j}.png")
            cv2.imwrite(word_path, word)

print("Segmentation complete.")