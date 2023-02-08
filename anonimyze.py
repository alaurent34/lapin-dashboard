import os
import pandas as pd
from tkinter import Tk
from tkinter.filedialog import askopenfilename

SAMPLE = None

if __name__=='__main__':

    Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
    data_file = askopenfilename(filetypes=[('Fichier de données', '*.csv')],
                                            initialdir=os.path.join(os.getcwd(), '../lapin/output'),
                                            title="Select file", multiple=False) # show an "Open" dialog box and return the path to the selected file

    if data_file:
        df = pd.read_csv(data_file)
        plaque_unique = df[['plaque']].drop_duplicates().reset_index(drop=True).reset_index().set_index('plaque').to_dict()['index']
        df['plaque'] = df['plaque'].map(plaque_unique)
        if SAMPLE:
            df = df.sort_values('datetime').head(SAMPLE)
        df.to_csv('./data/raw_data_project.csv', index=False)
        pd.DataFrame.from_dict({'mapping':plaque_unique}).to_csv('./data/mapping_plaque_anon.csv', index_label='plaque')