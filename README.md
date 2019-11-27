# image_loader
function to read a local set of images for machine learning image datasets on disk

# enter the path containing the folder of the folders containing the images
def load_images(path=r"..\datasets\images\notmnist\*\*"):
#     import dependencies
    import glob
    import matplotlib.image as mpimg
    from sklearn.preprocessing import LabelEncoder
    data={"data":[],"targets":[],"labels":[]}
#     using regular expressions to search for subfolders two layers deep
    for file_name in glob.glob(path):
        if file_name[-3:]=="png": #make sure you are only reading pictures or files ending with png
            target=file_name[28:29] #slice out the label name(this is partiular to this layout)

            try:
                image=mpimg.imread(file_name)#try converting the image to a numpy array
            except:
                print("there was an error parsing {}".format(file_name))

            data['data'].append(image)
            data['targets'].append(target)
            if target not in data['labels']:
                data['labels'].append(target)
    data['targets']=LabelEncoder().fit_transform(data['targets'])
    return data
