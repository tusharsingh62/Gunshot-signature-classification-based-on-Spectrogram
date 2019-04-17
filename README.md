# Gunshot-signature-classification-based-on-Spectrogram

**Concept of best spectrogram generation for classification purpose**

Muzzle blast and Shockwave classification in gunshots based on spectrogram required some manual engineering to generate the dataset from raw signal. We have already extracted muzzle blast and shockwave using  Events_sepfrmpq function in matlab code or Python from Ground reflections removal in Gunshot signals repository from raw signals (Experimentally collected data & Mp3 files of gunshots)

Now dataset is generated as spectrogram image and labeled accordingly in Muzzle balst or Shockwave. We are more concered about energy content in the signature. Shockwave signature is more sharp and less spreaded in time domain i.e high frequency content must be present. Muzzle blast is more spreaded in time domain i.e less frequency content must be there.

For example, I have uploded spectrogram sample for both shockwave and muzzle blast. We can quickly diffrentiate just by looking in the samples.

Now, we have a dataset in the form of image, and it becomes image classification problem. I used to create convolutional neural network architecture since we have small dataset, it is necessary to build simple architecture for better accuray in test set.
Keras library is used to create convolutional and pooling layers

Finally, achieved 100% accuracy in 27 Epochs for test data.
