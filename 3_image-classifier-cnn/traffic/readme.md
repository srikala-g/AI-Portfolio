Scenarios

1. Epoch - 10, Dropout size = 0.5 , Single convolutional + Max pooling layer - Test evaluation - 333/333 - 1s - 2ms/step - accuracy: 0.9502 - loss: 0.1940
2. Epoch - 50, Single convolutional + Max pooling layer -  Test evaluation -  333/333 - 1s - 2ms/step - accuracy: 0.9850 - loss: 0.1055
   [Observation: Increaase in Epoch alone improved accuracy and reduced loss function]
3. Epoch - 100, Single convolutional + Max pooling layer -  Test evaluation -  333/333 - 1s - 2ms/step - accuracy: 0.9855 - loss: 0.0950
   [Observation: Further increaase in Epoch alone showed marginal improvement in accuracy and reduction in loss function]
4. Epoch - 10, Two convolutional + Max pooling layer - Test evaluation - 333/333 - 1s - 3ms/step - accuracy: 0.9862 - loss: 0.0651
   [Observation: Epoch - 10 but two layers of Con + Max showed better accuracy and reduction in loss function compared to increasing just Epoch as seen in test #2]
5. Epoch - 50, Two convolutional + Max pooling layer - Test evaluation -333/333 - 1s - 3ms/step - accuracy: 0.9930 - loss: 0.0394
   [Observation: Increasing the Epoch further increase accuracy and reduced loss function. But should avoid overfitting]
6. Epoch - 10, change  filter 4, 4. One convolutional + Max pooling layer - Test evaluation -  333/333 - 0s - 1ms/step - accuracy: 0.9733 - loss: 0.1079
   [Observation: Changed filter, reduced accuracy and increaase loss function]
7. Epoch - 10, change  filter 2, 2, One convolutional + Max pooling layer - Test evaluation - 333/333 - 0s - 1ms/step - accuracy: 0.9306 - loss: 0.2940
   [Observation: Changed filter, reduced accuracy and increaase loss function]
8. Epoch - 10, change size of 32 to 50 filter 3, 3, One convolutional + Max pooling layer - Test evaluation -  333/333 - 0s - 951us/step - accuracy: 0.9712 - loss: 0.1428
   [Observation: Changed size of filter, reduced accuracy and increaase loss function]
9. Epoch - 10, change max pool size - 3,3, One convolutional + Max pooling layer - Test evaluation -  333/333 - 0s - 1ms/step - accuracy: 0.9668 - loss: 0.1463
   [Observation: Changed max pool size performance marginally better than #1]
9. Epoch - 10, change max pool size - 3,3, Two convolutional + Max pooling layer - Test evaluation - 333/333 - 0s - 1ms/step - accuracy: 0.9449 - loss: 0.1813
   [Observation: Changed max pool size performance lower than #4]
10. Epoch - 10, Dropout - 0.2, change max pool size - 2,2, One convolutional + Max pooling layer - Test evaluation - 333/333 - 1s - 2ms/step - accuracy: 0.9743 - loss: 0.1174
   [Observation: Reduced Dropout size, performance better than #1]
   Epoch - 10, Dropout - 0.2, change max pool size - 2,2, Two convolutional + Max pooling layer - Test evaluation - 333/333 - 1s - 3ms/step - accuracy: 0.9831 - loss: 0.0634
   [Observation: Reduced Dropout size, two con + max layer performance better than #10, and similar to performance #12, #4]
11. Epoch - 10, Dropout - 0.7, change max pool size - 2,2, One convolutional + Max pooling layer - Test evaluation - 333/333 - 1s - 2ms/step - accuracy: 0.9277 - loss: 0.3444
   [Observation: Increased Dropout size , performance worse than #1]
12. Epoch - 10, Dropout - 0.7, change max pool size - 2,2, Two convolutional + Max pooling layer - Test evaluation - 333/333 - 1s - 3ms/step - accuracy: 0.9824 - loss: 0.0744
   [Observation: Increased Dropout size , two con+max layer performance improved comapared to #11 but marginally lower than #5]

   Epoch - 1-50, Two convolution and max-pool layers, with Dropout = 0.5 and filter size of 32,3,3 and max_pool size of 2,2 gives a reasonable performance

13. Wrote a Streamlit application to load an image and classify the same. The model was able to classify a few images correctly but had issues with a few signs and blurry images
14. Tested image from internet - cropped and resized - but the model was unable to classify it correctly. It could be an issue with the image

