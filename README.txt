NOTE:
we use a dataset downloaded from kaggle. Running any code which needs the dataset for the first time will
download the data to your computer and put it in your PC's cache folder.

HOW TO RUN:
Run project.ipynb for a walkthrough demonstration of different areas of our code.
To run the Eclass model on the biomarkers, from the project root run `python -m src.train_and_eval`
Similarly, to run the CNN on the images for a baseline (does not use biomarkers) run `python -m src.cnn`
Note that we precomputed the biomarker results on all images as doing so takes a long time, the results are in
features.csv and test_features.csv. Deleting these files and running train_and_eval will recompute them and generate
new csv files.

project.ipynb              - Notebook comparing CNN vs biomarker models.
features.csv               - Biomarker dataset computed from training images.
test_features.csv          - Biomarker dataset computed from validation images.

models/bm_nn.pt            - Trained biomarker neural network weights.
models/bm_rf.pkl           - Trained Random Forest biomarker model.
models/bm_svm.pkl          - Trained SVM biomarker model.
models/cnn_melanoma.pt     - Trained CNN weights for image-based classifier.

src/b.py                   - Computes B-series (blue-channel) biomarkers.
src/biomarker_models.py    - Defines NN/RF/SVM biomarker models and loaders.
src/cnn.py                 - CNN architecture, training, and evaluation on images.
src/compute_biomarkers.py  - Generates all biomarkers and writes features.csv.
src/g.py                   - Computes G-series (green-channel) biomarkers.
src/lesion_mask.py         - Computes binary lesion masks from input images.
src/mc.py                  - Computes MC-series (multichannel/color) biomarkers.
src/r.py                   - Computes R-series (red-channel) biomarkers.
src/train_and_eval.py      - Trains biomarker models and ensemble, saves checkpoints.
