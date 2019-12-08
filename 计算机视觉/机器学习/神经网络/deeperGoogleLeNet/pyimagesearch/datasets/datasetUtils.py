# import the necessary packages
import zipfile
import tarfile
import requests
import fnmatch
import shutil
import os

class DatasetUtils:
    def __init__(self, fileURL):
        self.url = fileURL        
    
    def downloadDataset(self):        
        # set file name as last / from url
        fileName = self.url.rsplit('/', 1)[-1]

        # change to dataset folder
        os.chdir('datasets')

        if (os.path.isfile(fileName)):
            print("[INFO] %s have been downloaded" % fileName)
        else:        
            # download the file contents in binary format
            print("[INFO] Download Dataset...")
            r = requests.get(self.url)

            # open method to open a file on your system and write the contents       
            with open(fileName, "wb") as code:
                code.write(r.content)

        filedir = os.path.join(os.getcwd(), fileName)
        file_tar, file_ext = os.path.splitext(filedir)

        print("[INFO] extracting %s... " % fileName) 
        if file_ext == ".gz" or file_ext == ".tgz":
            tar = tarfile.open(fileName)
            tar.extractall(path=os.getcwd())
            tar.close()
            print("[INFO] Completed download Dataset and extract into Dataset folder.")

    def processFlowers17Image(self):

        print("[INFO] process Flowers17 Dataset...")

        # get the class label limit
        class_limit = 17
        imagePerClass = 80

        # flower17 class names
        class_names = ["daffodil", "snowdrop", "lilyvalley", "bluebell", "crocus", 
            "iris", "tigerlily", "tulip", "fritillary", "sunflower", 
            "daisy", "coltsfoot", "dandelion", "cowslip", "buttercup", 
            "windflower", "pansy"]

        # create directory
        outdir = os.path.join(os.getcwd(), 'flowers17')
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        # input directory
        indir = os.path.join(os.getcwd(), 'jpg')
        
        # create filelist for each class

        # loop over the class labels
        nextClassImageIdx = 0
        for Idx in range(len(class_names)):
            classDir =  os.path.join(outdir, class_names[Idx])  
            if not os.path.exists(classDir):
                print("[INFO] create class folder: %s" %class_names[Idx])
                os.makedirs(classDir)
                idxImage = nextClassImageIdx
                for fileName in fnmatch.filter(os.listdir(indir), '*.jpg'):
                    shutil.move(os.path.join(indir, fileName), os.path.join(classDir, fileName))                   
                    idxImage += 1
                    if (idxImage >= (nextClassImageIdx + imagePerClass)):
                        nextClassImageIdx += imagePerClass 
                        break
