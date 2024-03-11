import pandas as pd

# To display all columns
pd.set_option('display.max_columns', None)

# To display all rows
pd.set_option('display.max_rows', None)

# To display the entire contents of each cell (useful for large strings)
pd.set_option('display.max_colwidth', None)


class TextFileParser:
    def __init__(self, txt_file):
        self.txt_file = txt_file
        with open(self.txt_file, 'r') as f:
            lines = f.readlines()
        self.data = [line.strip().split(',') for line in lines]

    def get_data(self):
        return self.data

def text_files_to_df(txt_files):
    names = []
    descriptions = []
    xmin = []
    ymin = []
    xmax = []
    ymax = []

    for txt_file in txt_files:
        parser = TextFileParser(txt_file)
        data = parser.get_data()
        for item in data:
            names.append(item[0])
            descriptions.append(item[1])
            xmin.append(int(item[2]))
            ymin.append(int(item[3]))
            xmax.append(int(item[4]))
            ymax.append(int(item[5]))

    df = pd.DataFrame({
        'image_id': names,
        'description': descriptions,
        'xmin': xmin,
        'ymin': ymin,
        'xmax': xmax,
        'ymax': ymax,
    })

    return df

# Define the path to your text file
txt_file_path = "pix_2_seq/data/annotations_train.txt"

# Create a list of text files (in this case, just one file)
txt_files = [txt_file_path]

# Define a dictionary to map labels to filenames
label_mapping = {
    'crazing': 0,
    'scratches': 1,
    'rolled-in_scale': 2,
    'pitted_surface': 3,
    'patches': 4,
    'inclusion': 5
}

# Build the DataFrame
df = text_files_to_df(txt_files)

# Add the 'label' column based on the filenames with hyphens, assigning 2 as the default label for unknown labels
df['label'] = df['image_id'].apply(lambda x: label_mapping.get(x.split('_')[0], 2))

# Add the 'image_path' column using a placeholder value (you should replace this with the actual image path)
df['image_path'] = 'pix_2_seq/data/IMAGES' + df['image_id']

# Reorder the columns to match your desired format
df = df[['image_id', 'description', 'xmin', 'ymin', 'xmax', 'ymax', 'label', 'image_path']]

# Save the DataFrame to an Excel file
df.to_excel("output.xlsx", index=False)
