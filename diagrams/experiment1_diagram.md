graph TD
    subgraph Data Preparation
        A1[Raw Dataset (Spambase / Lung Cancer)] --> A2[Label Encoding (if needed)]
        A2 --> A3[Train-Test Split (70% Train, 30% Test)]
        A3 --> A4[Normalization (StandardScaler)]
        A4 --> A5[PCA - Dimensionality Reduction]
        A5 --> A6[X_train_pca, X_test_pca, y_train, y_test]
    end

    subgraph Model Training
        A6 --> B1[Initialize LinearSVC (Scikit-learn)]
        B1 --> B2[Train SVM on X_train_pca, y_train]
        B2 --> B3[Model: Trained Classical Linear SVM]
    end

    subgraph Prediction & Evaluation
        B3 --> C1[Predict y_test_pred from X_test_pca]
        B3 --> C2[Predict y_train_pred from X_train_pca]
        C1 --> D1[Compute Accuracy, Precision, Recall, F1-score]
        C2 --> D2[Compute Training Accuracy]
        D1 --> D3[Calculate Generalization Gap]
        D2 --> D3
    end
