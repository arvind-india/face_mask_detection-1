# Rasperry Pi Mask Detection

Deploying the mask detection project is very easy. 
1. SCP the _rpi_object_detection_v3.py_ file and the following files to your Raspberry Pi:
* MobileNetSSD_deploy.caffemodel (person detection model)
* MobileNetSSD_deploy.prototxt.txt (person detection prototxt)

This is the model for the person detection which will run on the Rpi. 

2. Connect the Rpi camera to your Rpi. Connect a DSLR to your Rpi.
3. Ensure gphoto2 is properly installed http://www.gphoto.org/. This will give you control of your DSLR from the Rpi
4. Run _rpi_object_detection_v3.py_ from the Rpi. Ensure the proper args are in place
5. At the end of each day, transfer assets from your camera to your main workstation. Assets are to be placed in a folder for the current date and then processed using _data_processor.py_

## Notes
I used the [Rpi camera](https://www.raspberrypi.org/products/camera-module-v2/) to detect when a person is in frame, then trigger a secondary camera to record a video. I used a secordary camera because the image quality is much better. It is certainly possible to modify the script and just record frames from a camera thats also running the person detector, such as a USB camera or the [newest Rpi camera](https://www.raspberrypi.org/products/raspberry-pi-high-quality-camera/)


### Improvements
It would be preferable to automatically transfer assets from the Rpi / DSLR to your main workstation. I had to do many things localy, so that piece is not built out yet  
 

 

 