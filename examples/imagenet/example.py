import feature_extract as fe

def run_alexnet():
    alexnet = fe.CaffeFeatureExtractor(
            model_path="alexnet_deploy.prototxt",
            pretrained_path="alexnet.caffemodel",
            blob="fc6",
            crop_size=227,
            meanfile_path="imagenet_mean.npy"
            )
    fe.create_dataset(net=alexnet, datalist="train.txt", dbprefix="alexnet_train")
    fe.create_dataset(net=alexnet, datalist="test.txt", dbprefix="alexnet_test")

def run_vgg16_fc7():
    vgg16 = fe.CaffeFeatureExtractor(
            model_path="vgg16_deploy.prototxt",
            pretrained_path="vgg16.caffemodel",
            blob="fc7",
            crop_size=224,
            mean_values=[103.939, 116.779, 123.68]
            )
    fe.create_dataset(net=vgg16, datalist="train.txt", dbprefix="vgg16_fc7_train")
    fe.create_dataset(net=vgg16, datalist="test.txt", dbprefix="vgg16_fc_7test")

def run_vgg16_fc6():
    vgg16 = fe.CaffeFeatureExtractor(
            model_path="vgg16_deploy.prototxt",
            pretrained_path="vgg16.caffemodel",
            blob="fc6",
            crop_size=224,
            mean_values=[103.939, 116.779, 123.68]
            )
    fe.create_dataset(net=vgg16, datalist="train.txt", dbprefix="vgg16_fc6_train")
    fe.create_dataset(net=vgg16, datalist="test.txt", dbprefix="vgg16_fc6_test")

def run_googlenet():
    googlenet = fe.CaffeFeatureExtractor(
            model_path="googlenet_deploy.prototxt",
            pretrained_path="googlenet.caffemodel",
            blob="pool5/7x7_s1",
            crop_size=224,
            mean_values=[104.0, 117.0, 123.0]
            )
    fe.create_dataset(net=googlenet, datalist="train.txt", dbprefix="googlenet_train")
    fe.create_dataset(net=googlenet, datalist="test.txt", dbprefix="googlenet_test")

if __name__ == "__main__":
    run_alexnet()
    run_vgg16_fc7()
    run_vgg16_fc6()
    run_googlenet()
