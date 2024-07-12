import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Container, Typography, Box, Divider, Grid, Button } from '@mui/material';
import articleImage from '../assets/articleImage.png';
import ArrowBackIcon from "@mui/icons-material/ArrowBack";

const Article = () => {
    const navigate = useNavigate();
  return (
    <Container sx={{ py: 8 }}>
      <Box display="flex" justifyContent="space-between" alignItems="center">
        <Typography variant="h4" gutterBottom>
          Writer Age Classification from Handwritten Documents
        </Typography>
      <Button
        variant="contained"
        startIcon={<ArrowBackIcon />}
        onClick={() => navigate(-1)}
        sx={{ mb: 2 }}
      >
        Back
      </Button>
      </Box>
      <Grid container spacing={4} alignItems="flex-start">
        <Grid item xs={12} md={4}>
          <img src={articleImage} alt="Article Illustration" style={{ width: '100%', borderRadius: '8px' }} />
        </Grid>
        <Grid item xs={12} md={8}>
          <Typography variant="h6" gutterBottom>
            Group Number: BS-SE-24-209
          </Typography>
          <Typography variant="h6" gutterBottom>
            Participants: Ofri Rom, Tomer Damti, Maayan Rabinovitch
          </Typography>
          <Typography variant="h6" gutterBottom>
            Academic Advisors: Dr. Irina Rabaev, Dr. Marina Litvak
          </Typography>
          <Typography variant="h6" gutterBottom>
            Institution: SCE - Shamoon College of Engineering, Be'er Sheva, 2024
          </Typography>
        </Grid>
      </Grid>
      <Divider sx={{ my: 4 }} />
      <Box mt={4}>
        <Typography variant="h5" gutterBottom>
          1. Background
        </Typography>
        <Typography paragraph>
          Handwriting is a window into the inner world of the writer. Beyond the written words, its shape and nature reflect the writer's personality, age, and emotional state.
        </Typography>
        <Typography paragraph>
          For many years, researchers have tried to decipher the code of handwriting to learn about the person who created it. Over the years, different approaches have been developed, new technologies have been tested, but many questions remain open.
        </Typography>
        <Typography paragraph>
          One of the most fascinating topics in this field is the identification of the writer's age. Can we determine the age of a person based solely on their handwriting?
        </Typography>
        <Typography paragraph>
          The challenge: Identifying age based on handwriting is not a simple task. Many factors influence handwriting, including geographical location, culture, education, and even momentary mood.
        </Typography>
        <Typography paragraph>
          The opportunity: With technological advancement, new ways to explore and analyze handwriting are emerging. Innovative algorithms, using "Deep Learning", offer significant potential for improving age recognition accuracy. Additionally, there is a need for dedicated research in Hebrew handwriting, as most studies have focused on English and Arabic.
        </Typography>
        <Divider sx={{ my: 4 }} />
      </Box>
      <Box mt={4}>
        <Typography variant="h5" gutterBottom>
          2. Motivation
        </Typography>
        <Typography paragraph>
          The field of age identification based on handwriting has gained momentum in recent years, thanks to technological advances and its vast potential.
        </Typography>
        <Typography paragraph>
          New models based on deep neural networks (CNNs), such as VGGNet, ResNet, Inception, and Xception, demonstrate impressive capabilities in classifying images and videos. However, there is still a significant need to improve the accuracy of age identification based on handwriting, especially in Hebrew.
        </Typography>
        <Typography paragraph>
          The current project aims to investigate and examine the potential of new and improved CNN models for age classification based on Hebrew handwriting. Our motivation is to develop models whose age range results will enable better decision-making in various fields.
        </Typography>
        <Divider sx={{ my: 4 }} />
      </Box>
      <Box mt={4}>
        <Typography variant="h5" gutterBottom>
          3. Problem Definition
        </Typography>
        <Typography paragraph>
          Identifying age and gender based on handwriting remains a complex challenge for both humans and computer systems. Handwriting contains a vast amount of information, such as letter shapes, writing pressure, writing speed, and more. These factors make it difficult to accurately represent gender and age features.
        </Typography>
        <Typography paragraph>
          Many studies have been conducted on this topic, but a high level of accuracy has not yet been achieved. Most studies have focused on Latin and Arabic scripts. There is a need for dedicated databases in Hebrew.
        </Typography>
        <Typography paragraph>
          Additional complexities: Handwriting is influenced by many factors, including age, gender, culture, education, mood, and even writing instruments. Technological advancements open new possibilities for age identification from handwriting, and we are very optimistic about this.
        </Typography>
        <Divider sx={{ my: 4 }} />
      </Box>
      <Box mt={4}>
        <Typography variant="h5" gutterBottom>
          4. Literature Review
        </Typography>
        <Typography variant="h6" gutterBottom>
          4.1. Introduction
        </Typography>
        <Typography paragraph>
          Handwriting is a fascinating phenomenon with a long history. Since ancient times, humans have used handwriting to communicate, document events, share knowledge, tell stories, and express emotions. Handwriting reflects not only the written words but also the writer's personality, emotional state, health, age, and even culture.
        </Typography>
        <Typography paragraph>
          Handwriting has received much attention throughout history from both artists and researchers. Many artists saw handwriting as a unique expression of creativity and emotion, using it in their works. Researchers, on the other hand, have tried to explore the connection between handwriting and personality, psychological traits, and even health status.
        </Typography>
        <Typography paragraph>
          In this project, we will align ourselves with the latter group and attempt to examine the connection between handwriting and the writer's age using machine learning tools.
        </Typography>
        <Typography variant="h6" gutterBottom>
          4.2. Importance and Applications
        </Typography>
        <Typography paragraph>
          Identifying the writer's age based on a handwriting image is of great importance in many fields, including:
        </Typography>
        <Typography paragraph>
          Criminal investigations: Handwriting analysis of suspects can provide important clues about their age, emotional state, and personality. Handwriting can also be used to compare different handwriting samples and identify connections between suspects. Identifying the writer's age from handwriting can help in investigating cases of document forgery, theft, and fraud.
        </Typography>
        <Typography paragraph>
          Analysis of historical manuscripts: Estimating the writer's age can contribute to dating ancient manuscripts. This can help in understanding historical events and social processes. Additionally, analyzing the handwriting of historical figures can provide new information about their personality, emotions, and mental state.
        </Typography>
        <Typography paragraph>
          Diagnosing learning disabilities: Handwriting can be used as a diagnostic tool for identifying dysgraphia, a learning disability characterized by difficulty in writing. Analyzing textural features, letter shapes, and handwriting flow can help in identifying this and other learning disabilities.
        </Typography>
        <Typography paragraph>
          Identifying neurodegenerative diseases: Handwriting can serve as a diagnostic tool for identifying neurodegenerative diseases such as Alzheimer's and Parkinson's. Analyzing handwriting features can help in detecting early signs of these diseases. Identifying the writer's age from handwriting can contribute to assessing disease progression and planning appropriate treatment.
        </Typography>
        <Typography paragraph>
          Other fields: Age identification from handwriting can be used in other fields such as graphology, archaeology, art, and more. Developing new technologies for age identification from handwriting opens new possibilities for research and application.
        </Typography>
        <Typography variant="h6" gutterBottom>
          4.3. Research Challenges
        </Typography>
        <Typography paragraph>
          Identifying the writer's age from their handwriting is not a simple task at all. A person's handwriting is influenced by many factors: age, education, culture, writing instrument, emotional state, and even health. As a result, it is difficult to generalize handwriting features across a broad population.
        </Typography>
        <Typography paragraph>
          Several challenges may be encountered: A person may write differently depending on the context and purpose of the writing, and environmental factors such as lighting and noise also influence handwriting. Blurred or unclear images make it difficult to analyze handwriting. Furthermore, different shooting angles, uneven lighting, and image noise can affect the accuracy of feature identification.
        </Typography>
        <Typography paragraph>
          Developing effective models for identifying the writer's age requires a large and diverse dataset of handwriting images labeled by age groups. Collecting such a dataset requires time and resources and may be biased as it may not include equal representation of all population groups.
        </Typography>
        <Typography paragraph>
          Developing machine learning models for identifying the writer's age is a complex task. These models need to handle different writing styles between writers, poor image quality, and limited dataset size. There is also the "black box" problem, making it difficult to understand how decisions are made by these models.
        </Typography>
        <Typography paragraph>
          It is important to use this technology responsibly to prevent discrimination or harm to people and protect their privacy. People should be informed about how this technology is used and their consent obtained before analyzing their handwriting.
        </Typography>
        <Typography paragraph>
          Despite the many challenges, identifying the writer's age is a research area with immense potential. Developing effective models can significantly contribute to fields such as criminal investigations, analysis of historical manuscripts, and diagnosis of learning disabilities. Research in this area is continually advancing, and it may one day be possible to easily and accurately identify a writer's age based on their handwriting, considering all influencing factors.
        </Typography>
        <Typography paragraph>
          The road ahead is long, but we are heading in the right direction. Continued development of new technologies, alongside in-depth research and attention to ethics, will allow us to make the most of writer age identification for the benefit of humanity.
        </Typography>
        <Typography variant="h6" gutterBottom>
          4.4. Existing Solutions
        </Typography>
        <Typography paragraph>
          Identifying the writer's age from handwriting is an active and developing research area with great potential in many fields. In recent years, many technological solutions have been developed, mainly based on machine learning and deep neural networks. This review presents several existing solutions, focusing on tools, datasets, and the level of accuracy achieved.
        </Typography>
        <Typography paragraph>
          Somaya Al Maadeed and Abdelaali Hassaine's study [2] presents a method for classifying handwriting by age, gender, and nationality, using geometric features, random forests, and Kernel Discriminant Analysis. The study is based on the QUWI dataset and shows accuracy ranging from 53.66% to 74.05% for the examined attributes. The study highlights the paramount importance of efficient feature extraction to distinguish between writers based on their handwriting.
        </Typography>
        <Typography paragraph>
          Irina Rabaev, Izadeen Alkoran, Odai Wattad, and Marina Litvak's study [4] presents a B-CNN system based on the ResNet model. The model aims to classify writers by gender and age based on their handwriting. The study uses English, Arabic, and Hebrew images. Due to the relatively low success of the VGG model, the researchers switched to using the ResNet model. They used the KHATT, QUWI, and HHD datasets.
        </Typography>
        <Typography paragraph>
          In gender classification, the researchers achieved accuracy ranging from 76% to 88.33% (for the KHATT, ICDAR 2013, HHD datasets), with ResNet often being the leading model (for HHD, which is significantly smaller than the others, Xception achieved slightly higher results).
        </Typography>
        <Typography paragraph>
          In age group classification performed on the KHATT dataset, the researchers recorded accuracy of 81.11% for two age groups, 67.30% for three age groups, and 66.65% for four age groups.
        </Typography>
        <Typography paragraph>
          Irina Rabaev, Marina Litvak, Sean Asulin, and Or Haim Tabibi's study [5] presents a CNN system used on the HHD, ICDAR 2013, ICDAR 2015 datasets.
        </Typography>
        <Typography paragraph>
          The model aims to classify writers by gender based on images of their handwriting. On the HHD dataset, the researchers achieved 85% accuracy using the Xception model. On the ICDAR 2013 dataset, they achieved 75% accuracy using the EfficientNet model. On the ICDAR 2015 dataset, they achieved accuracy ranging from 67% to 75% using EfficientNet and Xception, respectively.
        </Typography>
        <Typography paragraph>
          Mina Rahmanian and Mohammad Amin Shayegan's study [6] presents a system based on convolutional neural networks (CNNs) for classifying writers by gender and dominant hand based on their handwriting. The system uses deep learning algorithms applied to handwriting images to classify them by gender and dominant hand. The accuracy achieved is 93.75% for gender prediction and 92.59% for dominant hand prediction.
        </Typography>
        <Typography paragraph>
          Zhiheng Huang, Palaiahnakote Shivakumara, Maryam Asadzadeh Kaljahi, Ahlad Kumar, Umapada Pal, Tong Lu, and Michael Blumenstein's study [7] presents a method for estimating the writer's age using CNNs and geometric features. The system uses handwriting images and deep learning algorithms to extract relevant features and estimate the writer's age automatically. The accuracy achieved ranges from 80.89% to 82.38% using the Average Classification Rate (ACR) measure.
        </Typography>
        <Typography paragraph>
          Ahmed A. Elngar, Nikita Jain, Divyanshu Sharma, Himani Negi, Anuj Trehan, and Akash Srivastava's study [8] presents a system for predicting five personality traits. They work on a dataset they created called Personality Detection Dataset, which contains only 125 records. It is based on personality questionnaire results filled out by the participants. The researchers use two deep learning models: Personality Analyzing Network and PersonaNet. The accuracy achieved in the study ranges from 65% to 85% for different personality traits.
        </Typography>
        <Typography paragraph>
          Catherine Taleb, Laurence Likforman-Sulem, Chafic Mokbel, and Maha Khachab's study [9] presents a system for predicting Parkinson's disease (PD). The researchers work on a dataset they created called PDMultiMC, which contains handwriting, speech signals, and eye movement recordings, collected from 42 participants. The researchers combined the use of a CNN model and a CNN-BLSTM model. They achieved 97.62% accuracy for early detection of the disease.
        </Typography>
        <Typography paragraph>
          Mohamed Syazwan Asyraf Bin Rosli, Iza Sazanita Isa, Siti Azura Ramlan, Siti Noraini Sulaiman, and Mohd Ikmal Fitri Maruzuki's study [10] presents a CNN system for predicting dyslexia. The system is based on the LeNet-5 model with added layers to create an extended model. The researchers collected data from the NIST Special Database 19 and achieved 95.34% accuracy in predicting the disorder.
        </Typography>
        <Typography paragraph>
          Najla AL-Qawasmeh, Muna Khayyat, and Ching Y. Suen's study [11] presents a system based on an SVM model and an NN model. They used features such as Diagonal Irregularity (SI), Pen Pressure Irregularity (PPI), Text Line Irregularity (TLI), and Percentage of Black and White Pixels (PWB). The researchers worked on the KHATT and FSHS datasets and tried to create an age group prediction for two age groups. The study achieved a maximum accuracy of 71% using the SVM model on the FSHS dataset.
        </Typography>
        <Typography variant="h6" gutterBottom>
          4.5. Conclusions and Implications
        </Typography>
        <Typography paragraph>
          Existing solutions demonstrate the potential of machine learning and neural networks for identifying the writer's age based on their handwriting. It is evident that CNN is particularly effective for handwriting classification, both for age and other attributes. However, the accuracy of age prediction is still not high enough, and further research is needed in this field.
        </Typography>
        <Typography paragraph>
          The current review suggests possible approaches for age prediction from handwriting using machine learning technologies. The choice of model, features, and additional tools depends on the specific research goal. It is important to remember that the quality of the dataset greatly affects prediction accuracy, so careful attention should be given to collecting diverse and representative data.
        </Typography>
        <Typography paragraph>
          Continued research in the field of writer age identification is expected to lead to the development of more accurate and efficient models, overcoming challenges such as different writing styles, poor image quality, and limited dataset size. This development will enable widespread use of this technology in many fields, such as criminal investigations, analysis of historical manuscripts, and diagnosis of learning disabilities.
        </Typography>
        <Divider sx={{ my: 4 }} />
      </Box>
      <Box mt={4}>
        <Typography variant="h5" gutterBottom>
          5. Development Requirements
        </Typography>
        <Typography component="div">
          <ul>
            <li>
              <Typography variant="body1">
                <strong>Age Identification:</strong> The system should be able to identify the age of the person based on their handwriting.
              </Typography>
            </li>
            <li>
              <Typography variant="body1">
                <strong>Preprocessing of Handwriting Images:</strong> The system should be able to process handwriting from any person.
              </Typography>
            </li>
            <li>
              <Typography variant="body1">
                <strong>Real-Time Performance:</strong> The system should be able to operate in real-time.
              </Typography>
            </li>
            <li>
              <Typography variant="body1">
                <strong>System Accuracy:</strong> The system should be highly accurate.
              </Typography>
            </li>
            <li>
              <Typography variant="body1">
                <strong>System Efficiency:</strong> The system should be efficient and fast.
              </Typography>
            </li>
            <li>
              <Typography variant="body1">
                <strong>Experiments with Various Models:</strong> Training the system on different models to achieve the best results.
              </Typography>
            </li>
            <li>
              <Typography variant="body1">
                <strong>Model Preservation:</strong> Integration of pre-trained models.
              </Typography>
            </li>
            <li>
              <Typography variant="body1">
                <strong>Training on KHATT Dataset:</strong> Improving age identification accuracy by training models on the KHATT dataset.
              </Typography>
            </li>
            <li>
              <Typography variant="body1">
                <strong>Training on HHD Dataset:</strong> Improving age identification accuracy by training models on the HHD dataset.
              </Typography>
            </li>
            <li>
              <Typography variant="body1">
                <strong>Training on IAM Dataset:</strong> Improving age identification accuracy by training models on the IAM dataset.
              </Typography>
            </li>
          </ul>
        </Typography>
          <Divider></Divider>
      </Box>
      <Box mt={4}>
        <Typography variant="h5" gutterBottom>
          6. Datasets
        </Typography>
        <Typography paragraph>
          In this project, we use several unique handwriting datasets in Hebrew, Arabic, and English, focusing on identifying the writer's age. Our goal is to produce high-quality results and compare them to existing studies conducted on these datasets.
        </Typography>
        <Typography paragraph>
          The following datasets are used: HHD, KHATT, and IAM, allowing us to expose the model to a wide range of writing styles and cultures. The selected datasets correspond to the same age groups and display a similar image structure.
        </Typography>
        <Typography variant="h6" gutterBottom>
          HHD:
        </Typography>
        <Typography paragraph>
          Source: The HHD dataset was first introduced in the paper "Age Identification from Hebrew Handwriting Documents" by Rabaev, I., Barakat, B. K., Churkin, A., & El-Sana, J. [12] in 2020.
        </Typography>
        <Typography paragraph>
          Description: This dataset consists of documents written in Hebrew handwriting. Each document contains a paragraph from 50 categories, averaging 62 words per paragraph. The dataset includes handwriting from people of different backgrounds and age groups, ensuring diverse representation of writing styles. Age classification was performed manually and verified by three members of the research team.
        </Typography>
        <Typography variant="h6" gutterBottom>
          KHATT:
        </Typography>
        <Typography paragraph>
          Source: First introduced in the paper by Mahmoud, S. A., Ahmad, I., Alshayeb, M., Al-Khatib, W. G., Parvez, M. T., Fink, G. A., ... & El Abed, H. (2012, September) [13].
        </Typography>
        <Typography paragraph>
          Description: The KHATT dataset contains 5,000 paragraphs of handwritten Arabic text written by 1,000 individuals. Each writer contributed five paragraphs, including:
        </Typography>
        <Typography component="div">
          <ul>
            <li>
              <Typography paragraph>
                Two paragraphs randomly selected from 12 different categories (e.g., news, literature, sports).
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                Two minimal paragraphs covering all forms of Arabic letters.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                One paragraph of free text.
              </Typography>
            </li>
          </ul>
        </Typography>
        <Typography paragraph>
          In addition to the text, the dataset includes demographic information about each writer, such as gender, origin, and education. Age classification was also added, dividing the writers into four age groups:
        </Typography>
        <Typography component="div">
          <ul>
            <li>
              ≤15
            </li>
            <li>
              16-25
            </li>
            <li>
              26-50
            </li>
            <li>
              ≥51
            </li>
          </ul>
        </Typography>
        <Typography paragraph>
          This dataset is valuable for researchers in natural language processing, handwriting recognition, and Arabic text analysis.
        </Typography>
        <Typography variant="h6" gutterBottom>
          IAM:
        </Typography>
        <Typography paragraph>
          Source: First introduced by Marti, U. V. & Bunke, H (2002) [14].
        </Typography>
        <Typography paragraph>
          Description: The IAM dataset contains 1,066 handwritten forms in English written by approximately 400 different writers. The forms are based on the Lancaster-Oslo/Bergen (LOB) word corpus, which contains about one million words. In total, the dataset contains 82,227 words from a vocabulary of 10,841 words. The dataset consists of complete sentences in English and can serve as a basis for a wide range of handwriting recognition tasks. Its uniqueness lies in the possibility of incorporating linguistic knowledge beyond the dictionary level in the recognition process, as this knowledge can be automatically extracted from the dataset itself. Additionally, the dataset includes image processing tools for extracting the handwritten text from the forms and segmenting the text into lines and words.
        </Typography>
        <Typography paragraph>
          Key points:
        </Typography>
        <Typography component="div">
          <ul>
            <li>
              Access to a wide range of data allows us to train the model more effectively and tackle different handwriting recognition challenges.
            </li>
            <li>
              Comparing model results across multiple datasets helps strengthen the reliability of the findings and enables a more accurate performance evaluation.
            </li>
            <li>
              Using well-known datasets allows us to compare our research results with existing studies in the field and place our work in a broader context.
            </li>
          </ul>
        </Typography>
          <Divider></Divider>
      </Box>
      <Box mt={4}>
        <Typography variant="h5" gutterBottom>
          7. Models
        </Typography>
        <Typography paragraph>
          The success of the project largely depends on the correct choice and use of advanced Deep Learning models and algorithms. The model selection was made considering several factors, including accuracy, efficiency, and complexity. Running multiple experiments will allow us to choose the right model, which will produce accurate and efficient age classification based on handwriting.
        </Typography>
        <Typography paragraph>
          Models:
        </Typography>
        <Typography component="div">
          <ul>
            <li>
              VGGNet: A deep architecture with many small filters, known for its simplicity and good performance, but can be computationally expensive.
            </li>
            <li>
              ResNet: Introduces "skip connections" that allow information to flow directly through the network, helping to address the gradient vanishing problem and enabling deeper networks.
            </li>
            <li>
              Inception: Uses multiple filter sizes operating in parallel, capturing features at different scales within the image.
            </li>
            <li>
              Xception: Based on Inception, uses depth-wise separable convolutions for efficiency, achieving good accuracy with fewer parameters.
            </li>
            <li>
              EfficientNet: Focuses on achieving high accuracy with fewer parameters and computations compared to other models.
            </li>
          </ul>
        </Typography>
        <Typography paragraph>
          The project uses a variety of advanced deep learning models, including VGGNet, ResNet, Inception, Xception, and EfficientNet. Extensive experiments are conducted on all models to choose the ultimate model for accurate and efficient age classification based on handwriting.
        </Typography>
        <Typography variant="h6" gutterBottom>
          7.1. Detailed Description:
        </Typography>
        <Typography paragraph>
          <strong>Xception:</strong> Xception stands for "Extreme Inception" and is a deep convolutional neural network architecture inspired by the Inception architecture. It is known for its depth-wise separable convolutions, which aim to capture both spatial and channel-wise correlations in the data.
        </Typography>
        <Typography paragraph>
          <strong>Parameters:</strong> The number of parameters in Xception depends on the specific configuration, but it typically ranges from tens of millions to over a hundred million parameters.
        </Typography>
        <Typography paragraph>
          <strong>Layers:</strong> Xception consists of a series of convolutional, depth-wise separable convolutional, and pooling layers, followed by fully connected layers at the end for classification.
        </Typography>
          <Divider></Divider>
      </Box>
      <Box mt={4}>
        <Typography variant="h5" gutterBottom>
          8. Risk Management
        </Typography>
        <Typography component="div">
          <ul>
            <li>
              <Typography variant="body1">
                <strong>Low System Accuracy:</strong> Failure to meet project requirements. Mitigation: Improving feature extraction techniques, training more advanced models, using accuracy improvement techniques.
              </Typography>
            </li>
            <li>
              <Typography variant="body1">
                <strong>Failure to Find an Ideal Model:</strong> Receiving insufficient results. Mitigation: Conducting numerous experiments, considering the possibility of combining multiple models simultaneously.
              </Typography>
            </li>
            <li>
              <Typography variant="body1">
                <strong>Failure to Meet Deadlines:</strong> Delays in project execution. Mitigation: Detailed project planning, efficient time management, proper resource allocation.
              </Typography>
            </li>
            <li>
              <Typography variant="body1">
                <strong>Unexpected Technical Issues:</strong> Delays in project execution. Mitigation: Regular file backups, using Agile development methods, having backup plans.
              </Typography>
            </li>
            <li>
              <Typography variant="body1">
                <strong>Lack of Knowledge and Experience:</strong> Affecting project quality. Mitigation: Learning and practicing relevant technologies, consulting experts, searching for online information and support.
              </Typography>
            </li>
            <li>
              <Typography variant="body1">
                <strong>Insufficient Computing Capabilities:</strong> Delays in project execution and model training. Mitigation: Securing additional resources beyond what the college offers, using GPU components for training.
              </Typography>
            </li>
            <li>
              <Typography variant="body1">
                <strong>Incorrect Preprocessing:</strong> Not obtaining the desired result. Mitigation: Adapting the preprocessing process for identification and prevention.
              </Typography>
            </li>
            <li>
              <Typography variant="body1">
                <strong>System Overload:</strong> System crash and shutdown. Mitigation: Using appropriate hardware along with load management mechanisms.
              </Typography>
            </li>
          </ul>
        </Typography>
          <Divider></Divider>
      </Box>
      <Box mt={4}>
        <Typography variant="h5" gutterBottom>
          9. Experiments and Results
        </Typography>
        <Typography variant="h6" gutterBottom>
          9.1. Preprocessing
        </Typography>
        <Typography paragraph>
          Loading images containing text: The images containing text are loaded along with a label file. They are loaded in parallel, resulting in two synchronized lists of images and labels.
        </Typography>
        <Typography paragraph>
          Converting to Grayscale and Inverse Binarization: The images are converted to Grayscale. Additionally, inverse binarization is performed for each pixel: points closer to black become white and vice versa. This process is performed using the adaptive threshold function of the CV library.
        </Typography>
        <Typography paragraph>
          Image Cropping: The images are cropped so that their new boundaries minimize the peripheral area that does not contain text.
        </Typography>
        <Typography paragraph>
          Patching: Each image is separated into equal-sized patches smaller than the original image. The different patches overlap by 50% on the X and Y axes. Each patch is assigned the same label as the original image.
        </Typography>
        <Typography paragraph>
          At the end of this stage, two new lists of images and labels are created, where the images are patches of the original images.
        </Typography>
        <Typography variant="h6" gutterBottom>
          9.2. Model Training
        </Typography>
        <Typography paragraph>
          During training, we create a model object with the following values:
        </Typography>
        <Typography paragraph>
          - Loss function: "categorical crossentropy"
        </Typography>
        <Typography paragraph>
          - Optimizer function: Adam with a learning rate of 0.001
        </Typography>
        <Typography paragraph>
          We rely on pre-trained models and train them on our data. As a result, our model consists of trained layers and new layers tailored to our data. The neurons in the new layers are initialized randomly. Therefore, training the entire system will damage the trained layers due to the random information transfer from the new layers.
        </Typography>
        <Typography paragraph>
          To combat this problem, we initially "freeze" the basic model layers, preventing them from learning new information. We then run epochs until the loss function begins to plateau, indicating that the derivative function increases very slowly. At this stage, the neurons in the new layers are influenced by the basic model layers.
        </Typography>
        <Typography paragraph>
          We then "unfreeze" the basic model layers, allowing them to learn and change the neuron values. We run epochs again until the loss function begins to plateau.
        </Typography>
        <Typography variant="h6" gutterBottom>
          9.3. Model Testing
        </Typography>
        <Typography paragraph>
          During model testing, we first run preprocessing on the test images, except for the patching stage.
        </Typography>
        <Typography paragraph>
          Each image is then divided into patches. The model makes predictions for all patches of the same image, with the most frequent prediction added to the prediction list. At the end of the test images, a list is created containing the most frequent prediction for each image.
        </Typography>
        <Typography paragraph>
          The prediction list is compared to the true labels of the images, resulting in a confusion matrix.
        </Typography>
        <Typography variant="h6" gutterBottom>
          9.4. Experiment Results on Datasets
        </Typography>
        <Typography paragraph>
          In this study, we evaluated the performance of several deep learning models for identifying the writer's age based on images of their handwriting in Hebrew and Arabic. The experiments were conducted on two datasets: HHD and KHATT.
        </Typography>
        <Typography paragraph>
          The following tables present the accuracy of each model on each dataset:
        </Typography>
        <Typography variant="h6" gutterBottom>
          HHD (for 4 age groups)
        </Typography>
        <Typography paragraph>
          <strong>Accuracy:</strong>
        </Typography>
        <Typography component="div">
          <ul>
            <li>Vgg16: 55.17%</li>
            <li>Vgg19: 56.89%</li>
            <li>Xception: 50.86%</li>
            <li>ResNet: 56.34%</li>
            <li>ConvNeXtXLarge: 61.21%</li>
            <li>EfficientNet: 55.17%</li>
          </ul>
        </Typography>
        <Typography paragraph>
          ConvNeXtXLarge: This model achieved an accuracy of 61.21% on the HHD dataset. This may be because its network structure is more suited to identifying key features in Hebrew handwriting. Additionally, it contains significantly more layers than the other models, likely providing an advantage.
        </Typography>
        <Typography variant="h6" gutterBottom>
          KHATT (for 4 age groups)
        </Typography>
        <Typography paragraph>
          <strong>Accuracy:</strong>
        </Typography>
        <Typography component="div">
          <ul>
            <li>Proposed (Vgg16): 61.07%</li>
            <li>Proposed (Vgg19): 65.76%</li>
            <li>Proposed (Xception): 64.42%</li>
            <li>Proposed (ResNet): 65.77%</li>
            <li>Rabaev, Litvak [4] (B-ResNet): 66.65%</li>
            <li>Proposed (ConvNeXtXLarge): 63.59%</li>
            <li>Proposed (EfficientNet): 65.10%</li>
          </ul>
        </Typography>
        <Typography paragraph>
          ResNet: This model achieved an accuracy of 65.77% on the KHATT dataset.
        </Typography>
        <Typography paragraph>
          It is notable that for both datasets, the performance of the Vgg19 and ResNet models was very similar.
        </Typography>
          <Divider></Divider>
      </Box>
      <Box mt={4}>
        <Typography variant="h5" gutterBottom>
          10. Bibliography
        </Typography>
        <Typography paragraph>
          <ol>
            <li>Alaei, F., & Alaei, A. (2023). Review of age and gender detection methods based on handwriting analysis. Neural Computing and Applications, 35(33), 23909-23925.</li>
            <li>Al Maadeed, S., & Hassaine, A. (2014). Automatic prediction of age, gender, and nationality in offline handwriting. EURASIP Journal on Image and Video Processing, 2014(1), 1-10.</li>
            <li>Mekyska, J., Faundez-Zanuy, M., Mzourek, Z., Galaz, Z., Smekal, Z., & Rosenblum, S. (2016). Identification and rating of developmental dysgraphia by handwriting analysis. IEEE Transactions on Human-Machine Systems, 47(2), 235-248.</li>
            <li>Rabaev, I., Alkoran, I., Wattad, O., & Litvak, M. (2022). Automatic gender and age classification from offline handwriting with bilinear ResNet. Sensors, 22(24), 9650.</li>
            <li>Rabaev, I., Litvak, M., Asulin, S., & Tabibi, O. H. (2021, September). Automatic gender classification from handwritten images: a case study. In International Conference on Computer Analysis of Images and Patterns (pp. 329-339). Cham: Springer International Publishing.</li>
            <li>Rahmanian, M., & Shayegan, M. A. (2021). Handwriting-based gender and handedness classification using convolutional neural networks. Multimedia Tools and Applications, 80, 35341-35364.</li>
            <li>Huang, Z., Shivakumara, P., Kaljahi, M. A., Kumar, A., Pal, U., Lu, T., & Blumenstein, M. (2023). Writer age estimation through handwriting. Multimedia Tools and Applications, 82(11), 16033-16055.</li>
            <li>Elngar, A. A., Jain, N., Sharma, D., Negi, H., Trehan, A., & Srivastava, A. (2020). A deep learning based analysis of the big five personality traits from handwriting samples using image processing. Journal of Information Technology Management, 12(Special Issue: Deep Learning for Visual Information Analytics and Management.), 3-35.</li>
            <li>Taleb, C., Likforman-Sulem, L., Mokbel, C., & Khachab, M. (2020). Detection of Parkinson’s disease from handwriting using deep learning: a comparative study. Evolutionary Intelligence, 1-12.</li>
            <li>Rosli, M. S. A. B., Isa, I. S., Ramlan, S. A., Sulaiman, S. N., & Maruzuki, M. I. F. (2021, August). Development of CNN transfer learning for dyslexia handwriting recognition. In 2021 11th IEEE International Conference on Control System, Computing and Engineering (ICCSCE) (pp. 194-199). IEEE.</li>
            <li>Najla, A. Q., Khayyat, M., & Suen, C. Y. (2023). Age detection from handwriting using different feature classification models. Pattern Recognition Letters, 167, 60-66.</li>
            <li>Rabaev, I., Barakat, B. K., Churkin, A., & El-Sana, J. (2020, September). The HHD dataset. In 2020 17th International Conference on Frontiers in Handwriting Recognition (ICFHR) (pp. 228-233). IEEE.</li>
            <li>Mahmoud, S. A., Ahmad, I., Alshayeb, M., Al-Khatib, W. G., Parvez, M. T., Fink, G. A., ... & El Abed, H. (2012, September). Khatt: Arabic offline handwritten text database. In 2012 International conference on frontiers in handwriting recognition (pp. 449-454). IEEE.</li>
            <li>Marti, U. V., & Bunke, H. (2002). The IAM-database: an English sentence database for offline handwriting recognition. International Journal on Document Analysis and Recognition, 5, 39-46.</li>
          </ol>
        </Typography>
      </Box>
    </Container>
  );
};

export default Article;
