import pandas as pd

def create_csv(csv):
    data_df = pd.read_csv(csv)
    classes = []
    is_inside = False

    for index, row in data_df.iterrows():
        is_inside = False
        cname = row['Label']
        for element in classes:
            if(cname == element):
                is_inside = True
        if (is_inside is False):
            print(classes)
            classes.append(cname)

    df = pd.DataFrame({'class_name': classes})
    df.to_csv('../data/classes.csv', index=False)

if __name__ == '__main__':
     create_csv('../data/combined_annotations.csv')