from charge import get_images
from hog import calculate_hog
from charge import save_var

# Charges the files to the workspace
train_images, test_images = get_images()

print('-------------- Start HOG --------------')
# Calculates HOG Descriptor for train and test images
hog_train = {}
hog_test = {}
for key in test_images.keys():
    print('--------------')
    print(key)
    for i in range(0, len(train_images[key])):
        train_act = calculate_hog(train_images[key][i])
        if key not in list(hog_train.keys()):
            hog_train[key] = [train_act]
        else:
            list_train = hog_train[key]
            list_train.append(train_act)
    print('Finished train')
    for j in range(0, len(test_images[key])):
        test_act = calculate_hog(test_images[key][j])
        if key not in list(hog_test.keys()):
            hog_test[key] = [test_act]
        else:
            list_test = hog_test[key]
            list_test.append(test_act)
    print('Finished test')

save_var('HOG_train.npy', hog_train)
save_var('HOG_test.npy', hog_test)
