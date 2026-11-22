# Notebook cell results

Screenshots of every cell of `notebooks/legal_simplification_lora.ipynb`, taken while running it on a free Google Colab T4 GPU. The files are numbered in the order of the notebook.

| File | Notebook section | What it shows |
|---|---|---|
| 01_mount_drive_and_clone_repo.png | Start | Mounts Google Drive and clones the GitHub repository if the folder is missing |
| 02_notebook_intro.png | Intro | Title and instructions |
| 03_gpu_check_T4.png | 0. Setup | CUDA available, GPU is a Tesla T4 |
| 04_install_libraries.png | 0. Setup | Library install (the red pip "dependency resolver" message is a harmless Colab warning) |
| 05_project_paths_and_imports.png | 0. Setup | Project folder found, `data_prep` and `evaluate` imported |
| 06_huggingface_token_from_colab_secrets.png | 0. Setup | Token read from Colab Secrets (it is never typed into the notebook) |
| 07_load_data_splits.png | 1. Data | 273 train / 58 validation / 60 test clauses |
| 08_load_base_model_code.png | 2. Base model | Qwen2.5-3B-Instruct loaded in 4-bit (QLoRA) |
| 09_load_base_model_progress.png | 2. Base model | Download and load finished, 2.06 GB of GPU memory used |
| 10_helper_functions.png | 3. Helpers | Chat formatting and greedy generation |
| 11_baseline_cell_code.png | 4. Baseline | Base model on the 60 test clauses |
| 12_baseline_outputs.png | 4. Baseline | Example base model outputs |
| 13_lora_key_ideas.png | 5. LoRA | Notes on LoRA and loss masking |
| 14_lora_hyperparameters_and_tokenization.png | 5. LoRA | Hyperparameters and tokenization with masked prompt |
| 15_attach_lora_trainable_parameters.png | 5. LoRA | 29,933,568 trainable parameters out of 3,115,872,256 (0.96%) |
| 16_training_arguments.png | 5. LoRA | Training arguments |
| 17_trainer_cell_code.png | 5. LoRA | Trainer cell (code only, see the note below) |
| 18_training_results_final_run.png | 5. LoRA | Final run: 54 steps in 06:30, loss per epoch, adapter saved |
| 19_loss_curve_code.png | 5. LoRA | Code for the loss curve |
| 20_loss_curve_plot.png | 5. LoRA | Training and validation loss |
| 21_save_training_log.png | 5. LoRA | Saves `results/train_log.json` |
| 22_finetuned_generation_code.png | 6. Fine-tuned | Generation with the fine-tuned model |
| 23_finetuned_outputs_part1.png | 6. Fine-tuned | Base vs fine-tuned outputs, first examples |
| 24_finetuned_outputs_part2.png | 6. Fine-tuned | Base vs fine-tuned outputs, last example |
| 25_evaluation_code.png | 7. Evaluation | Scoring with `src/evaluate.py` |
| 26_bertscore_model_download.png | 7. Evaluation | BERTScore model (roberta-large) download |
| 27_metrics_table_base_vs_finetuned.png | 7. Evaluation | Final metrics table |
| 28_error_analysis_code.png | 8. Error analysis | Code for the error analysis |
| 29_optional_reload_adapter_cells.png | Optional | Adapter files found, libraries installed |
| full_notebook_run_record.pdf | All | Whole run as one PDF, including the error-analysis output (the 5 worst fine-tuned clauses) and the base and fine-tuned output text |

**Note on file 17.** The original screenshot of this cell also showed a loss table from an earlier training run (06:47, final validation loss 1.809). That table does not belong to the final run, so the screenshot was cropped to the code. The final run (06:30, validation loss 1.796) is in file 18, in `results/train_log.json` and in the saved notebook output.
