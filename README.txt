project.ipynb              - Notebook comparing CNN vs biomarker models.
features.csv               - Tabulated biomarker dataset generated from images.

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

