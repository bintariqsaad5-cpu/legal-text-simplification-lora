# Web app results

Screenshots of the Gradio app (`app/app.py`) running on Google Colab. Each test pastes one clause that is not from the dataset and shows the base model and the fine-tuned model side by side.

| File | Test clause | Base model | Fine-tuned model |
|---|---|---|---|
| 00_launch_app_cell.png | The notebook cell that starts the app | | |
| 01_test_warranty_disclaimer.png | "As is" warranty disclaimer | Long but accurate | "the service is provided as is with no warranty." |
| 02_test_account_suspension.png | Account suspension if the user breaks the rules or commits fraud | Accurate, keeps the conditions | "the service can delete your account for any reason they see fit." (conditions lost) |
| 03_test_limitation_of_liability.png | Limitation of liability for indirect damages | Long and accurate | "the court of law limits liability for damages." (vague, misleading) |
| 04_test_changing_terms.png | Changing the terms, with notice and acceptance by continued use | Accurate | "the service can change its terms any time they see fit." (notice and acceptance lost) |

In these four tests the fine-tuned model is shorter and closer to the style of the training summaries, but it loses conditions in three of them. See section 7 of the main `README.md`.
