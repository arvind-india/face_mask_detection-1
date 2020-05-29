# face_mask_detection

## Data Processing

This section provides resources for gathering training data. In _real_time_video_capture.py_ we use the Rpi camera module to detect when a person is in frame, then trigger a secondary camera to record a video. We use a secordary camera because the image quality is much better. It is certainly possible to modify the script and just record frames from a camera thats also running the person detector.

It's also possible to just record video for an extended time and then filter after the fact. I was limited by storage space and couldnt do that. 

If you want to add in images from other sources (like google) to add more diversity in your data the _video_processing.py_ script can be leveraged for modifying the images for training
