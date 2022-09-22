 "706 Deep learning for sequence analysis" coursework @ City, University of London

# Follow these steps to reproduce my results:

1. Open this link: https://1drv.ms/u/s!ApqC2EDsdE7tjL80dm7PHZx8w8Vtxw?e=apejim

2. Right click "korpus90" and click "Download". See screenshot in "how_to_download_code.png" if in doubt.

3. Place the folder "korpus90" at the root of the zipped folder, i.e. in the same folder as the files reproduce_class.ipynb and reproduce_lemma.ipynb
Make sure that the downloaded folder is named "korpus90", otherwise the script will fail. 

4. Open reproduce_lemma.ipynb or reproduce_class.ipynb and run the cell. The former will reproduce the best model achieved when training on sequences of word lemmas. The latter will reproduce the best model achieved when trained on sequences of word class tokens. 




Below I describe what the main files do.

# run.py
This file should not be run. This file contains the code that trains and tests the model. The code calls on functions in preprocessing.py, dataset.py and gym.py

# gym.py
This file should not be run. This file contains functions that handles training and testing respectively

# preprocessing.py
This file should not be run. Contains functions that extracts sentences from the raw text data files and prepares it for the dataset class.

# dataset.py
This file should not be run. This file contains the custom dataset class as well as some helper functions related to building the dataloader. 

# models.py
This file should not be run. It contains the code that implements the models used in the coursework. 

# unique_words.ipynb
This file should not be run. It contains the code that was used to explore the vocabulary sizes and create the files that contain the vocabulary tokens. 

# device_param.py
This file should not be run. The code in this file determines whether training takes place on CPU or GPU. 
