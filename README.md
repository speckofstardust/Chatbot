# First Aid Chatbot with TensorFlow and NLTK

## Overview

This First Aid Chatbot is a useful tool for providing initial guidance on first aid measures for common injuries and health issues. The chatbot is built using TensorFlow for natural language processing and NLTK (Natural Language Toolkit) for language understanding and processing. The user interacts with the chatbot through a simple graphical interface created with tkinter.

## Features

- **First Aid Guidance:** Offers basic first aid information and guidance for common injuries and health concerns.
- **Interactive Chat Interface:** User-friendly tkinter-based chat interface for easy interaction with the chatbot.
- **TensorFlow-based Model:** Utilizes a TensorFlow model for natural language processing and understanding user queries.
- **NLTK for Language Processing:** Integrates NLTK for further language processing.

## Getting Started

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/speckofstardust/Chatbot
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download Kaggle Dataset:**
   Obtain the required dataset from Kaggle (here's what was used for this project: https://www.kaggle.com/datasets/therealsampat/intents-for-first-aid-recommendations) and place it in the appropriate directory.

4. **Train the Chatbot:**
   Run the training script to train the chatbot on the provided dataset.
   ```bash
   python train.py
   ```

5. **Run the Chatbot:**
   Launch the chatbot application using the following command.
   ```bash
   python chatbot.py
   ```

6. **Chat with the Bot:**
   Interact with the chatbot through the tkinter interface. Ask questions related to first aid, and the chatbot will provide guidance based on its training.

## Dataset

The dataset used for training this chatbot is sourced from Kaggle (https://www.kaggle.com/datasets/therealsampat/intents-for-first-aid-recommendations). The edited and formatted version of it for this project is uploaded here. It includes a diverse range of first aid scenarios to enhance the chatbot's knowledge and responsiveness.

## Improvements

The chatbot can be further improved by:

- Expanding the dataset with additional first aid scenarios.
- Enhancing the natural language processing model for better understanding.
- Adding multimedia features for a more interactive experience.

## Contributions

Contributions are welcome! If you have suggestions, find bugs, or want to contribute to the project, feel free to open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
