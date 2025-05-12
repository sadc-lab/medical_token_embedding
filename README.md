# Label Embedding for Self Attention

This repository is associated with the paper: [Are Medium-Sized Transformer Models Still Relevant for Medical Records Processing?](https://arxiv.org/abs/2404.10171).

## 📑 Table of Contents

- [How to Set Up the Environment](#how-to-set-up-the-environment)
- [Data Processing](#data-processing)
- [Training LESABert](#training-lesabert)
- [📄 License](#-license)
- [📚 Citation](#-citation)

## ⚙️ How to Set Up the Environment

1. **Create a new Python virtual environment** named `lesa_venv` using the `.yml` file by running the following command in your CLI:
    ```bash
    conda env create -f medical_lesa/lesa_venv.yml
    ```

2. **Activate the environment and create a Jupyter kernel** by running the following commands in the CLI:
    ```bash
    conda activate lesa_venv
    pip install ipykernel
    python -m ipykernel install --user --name lesa_venv --display-name "lesa_venv"
    ```

3. **Replace the `transformers/camembert` folder** with our version present in this repo by running the following command in your CLI:
    ```bash
    scp -r medical_lesa/camembert ~/.conda/envs/lesa_venv/lib/python3.11/site-packages/transformers/models
    ```

4. **Replace the `transformers/__init__.py` file** with our version present in this repo:
    ```bash
    scp -r medical_lesa/transformers__init__.py ~/.conda/envs/lesa_venv/lib/python3.11/site-packages/transformers/__init__.py
    ```

## 🧑‍⚕️ Data Processing

Before running the developed models or executing the `train.ipynb` notebook, users must first preprocess the medical data.

1. Begin by creating a folder named `sadcsip`. Inside this folder, place the two private files:
   - `NLP_data/2020.06.03_CHUSJ_Data_PatientID.csv`
   - `NLP_data/Labelling Le - 0 to 100.csv`

   These files contain the medical notes along with their corresponding annotations.

2. Next, open and run all cells in the `data_preprocessing.ipynb` Jupyter notebook. While the notebook is lengthy, it is fully commented and should be easy to follow.

Once completed, the required datasets will be generated in the correct format within the `sadcsip` folder.

## 🚀 Training LESABert

Now everything is ready to run the training process. Follow the instructions in the `train.ipynb` notebook for detailed steps on training the LESABert model with the preprocessed data.

## 📄 License

This project is licensed under the [MIT License](LICENSE).

You are free to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the software, provided that the original copyright and permission notice are included in all copies or substantial portions of the software.

## 📚 Citation

If you use this code or dataset in your research, please cite:

**Plain-text citation:**  
Lompo, A., & Le, T.-D. (2025). *medical\_token\_embedding: Preprocessing and Embedding of Medical Notes*. GitHub repository: https://github.com/aser97/medical_token_embedding

**BibTeX:**
```bibtex
@misc{lompo2025medicaltokenembedding,
  author       = {Aser Lompo and Thanh-Dung Le},
  title        = {medical_token_embedding: Preprocessing and Embedding of Medical Notes},
  year         = {2025},
  howpublished = {\url{https://github.com/aser97/medical_token_embedding}},
}
