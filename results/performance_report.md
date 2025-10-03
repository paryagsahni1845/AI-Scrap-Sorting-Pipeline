# Performance Summary: Real-Time Material Classification

## 1. Core Model Performance (Validation Set)

| Metric | Result | Interpretation |
| :--- | :--- | :--- |
| **Overall Accuracy** | **79.21%** | The model correctly classified the material type nearly 8 out of 10 times on unseen validation data. |
| **Precision (Macro)** | **78.05%** | When the model predicted a class, it was correct 78% of the time, averaged across all 6 classes. |
| **Recall (Macro)** | **76.72%** | The model successfully identified 76.7% of all instances for each class, averaged across all classes. |

---

## 2. Training Visuals

The Confusion Matrix highlights common misclassifications (off-diagonal values), revealing critical areas for future improvement:

| Major Misclassification Pairs | Error Count | Potential Cause |
| :--- | :--- | :--- |
| **Plastic $\rightarrow$ Glass** | **20** | Visual similarity of transparent/translucent items (e.g., plastic bottles looking like glass). |
| **Metal $\rightarrow$ Glass** | **14** | Reflective surfaces causing similar specular highlights, leading to confusion. |
| **Paper $\leftrightarrow$ Cardboard** | **5/4** | The model struggles to distinguish between flat paper products and thicker cardboard textures. |

**Key Takeaway:** The model is highly effective for `Paper` and `Glass` classes but requires targeted data augmentation (e.g., blurring, brightness variation) or collecting more varied samples to reduce the significant confusion between **Plastic** and **Glass/Metal** items.

### Training Convergence Plot

The training history confirms the effectiveness of the **ResNet-18 transfer learning approach**:

* **Rapid Improvement:** Both accuracy and loss curves show dramatic improvement within the first few epochs, demonstrating that the ImageNet pre-trained weights provided an excellent starting point.
* **Convergence:** The loss curves flatten out significantly by Epoch 8, indicating that the model has **converged**.
* **Generalization:** The validation loss remains stable and low, confirming the model achieved good **generalization** without severe overfitting, making the resulting model robust for the simulation environment.

---

## 3. Simulated Real-Time Performance & Review Log

The simulation successfully demonstrated deployment functionality:

| Performance Metric | Result |
| :--- | :--- |
| **Average Inference Time (ONNX)** | **~50-60 ms** |
| **Frame Rate Capacity** | $\approx 16-20$ FPS |
| **Low Confidence Threshold** | $\mathbf{0.85}$ |

### Review Summary (Active Learning Triage)

| Flag Type | Trigger Count (out of 50) |
| :--- | :--- |
| **Misclassified/Manual Flag** | **5** (Items where predicted $\ne$ true) |
| **Low Confidence Flag** | **23** (Items where confidence $< 0.85$) |
| **Total Items for Retraining Queue** | **28** (All items flagged above) |

**Conclusion:** The pipeline is fast and robust. The low confidence flags and misclassification checks successfully isolate challenging frames, providing a high-quality dataset of 28 hard examples for a future Active Learning retraining cycle.