import torch
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pandas as pd
import os

def append_df_to_csv(filename, df, **to_csv_kwargs):
    # Check if file exists and if yes, determine if the header should be written
    header = not os.path.exists(filename)
    
    # Append data to the CSV file
    df.to_csv(filename, mode='a', header=header, **to_csv_kwargs)

def draw_bbox_with_caption(image, bboxes, captions, color=(255, 0, 0), font_size=15):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()  # Use default font
    
    for i, caption in enumerate(captions):
        # Draw bbox if available; otherwise, default to a standard position for the caption
        if i < len(bboxes) and len(bboxes[i]) == 4:
            bbox = bboxes[i]
            draw.rectangle([(bbox[0], bbox[1]), (bbox[2], bbox[3])], outline=color, width=2)
            text_position = (bbox[0], bbox[1] - 10)
        else:
            # Default position could be the top left of the image, adjust as needed
            text_position = (10, i * (font_size + 5))  # Stacking captions vertically if no bboxes

        draw.text(text_position, caption, fill=color, font=font)

