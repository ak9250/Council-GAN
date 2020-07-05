import runway
import numpy as np
import tensorflow as tf
from PIL import Image
from test_gui import torch, Variable, transforms, face_recognition, encode_s, decode_s
import yaml


def get_config(config):
    with open(config, 'r') as stream:
        return yaml.safe_load(stream)


def run_net_work(img, entropy, config=config, use_face_locations=False, face_increes_by_dev_ratio=1.7, move_up_by_ratio=0):
    out_im_path = './tmp.jpg'
    in_im_path = './tmp_in.jpg'
    net_hight = config['crop_image_height']
    net_width = config['crop_image_width']
    net_new_size = config['new_size']
    do_pad_with_zeros_if_not_squared = True

    mean = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
    std = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
    transform_list = [transforms.ToTensor(), transforms.Normalize(mean=mean.tolist(), std=std.tolist())]

    transform_list = [transforms.CenterCrop((net_hight, net_width))] + transform_list
    transform_list = [transforms.Resize(net_new_size)] + transform_list
    transform = transforms.Compose(transform_list)

    img = np.array(img)

    # Find all the faces in the image using the default HOG-based model.
    # This method is fairly accurate, but not as accurate as the CNN model and not GPU accelerated.
    # See also: find_faces_in_picture_cnn.py
    if use_face_locations:
        # img = face_recognition.load_image_file(img_path)
        face_locations = face_recognition.face_locations(img=img, number_of_times_to_upsample=0)
        img_h = img.shape[0]
        img_w = img.shape[1]

        if do_pad_with_zeros_if_not_squared:
            # padd with zeros if the image is not squared
            left_index = 0
            up_index = 0
            if img_h != img_w:
                new_size = max(img_h, img_w)
                new_im = np.zeros((new_size, new_size, img.shape[2]))
                if img_h > img_w:
                    left_index = int((new_size - img_w)/2)
                    new_im[:, left_index:left_index+img_w, :] = img
                else:
                    up_index = int((new_size - img_h)/2)
                    new_im[up_index:up_index+img_h, :, :] = img
                for cur, face_location in enumerate(face_locations):
                    # (top, right, bottom, left)
                    face_location = (face_location[0]+up_index, face_location[1]+left_index, face_location[2]+up_index, face_location[3]+left_index)
                    face_locations[cur] = face_location
                img_h = new_size
                img_w = new_size
                old_img = img
                img = new_im.astype(np.uint8)

        final_res_img = transforms.ToTensor()(img)
        in_img = transforms.ToTensor()(img)

    else:
        # img = Image.open(img_path)
        # img = np.array(img)
        face_locations = [[0, img.shape[1], img.shape[0], 0]]
        img_h = img.shape[0]
        img_w = img.shape[1]
        final_res_img = transforms.ToTensor()(img)
        in_img = transforms.ToTensor()(img)

    for face_location in face_locations:
        top, right, bottom, left = face_location

        hight = bottom - top
        width = right - left
        if use_face_locations:
            # making the image larger because face_recognition  cuts the faces
            increes_by = int(max(hight, width) / face_increes_by_dev_ratio)

            if hight + increes_by > img_h or width + increes_by > img_w:
                # increes_by is too big
                increes_by_max_h = int((img_h - hight) / 2)
                increes_by_max_w = int((img_w - width) / 2)
                increes_by = min(increes_by_max_h, increes_by_max_w)

            top, right, bottom, left = top - increes_by, right + increes_by, bottom + increes_by, left - increes_by
            hight = bottom - top
            width = right - left

            if top < 0:
                top = 0
                bottom = hight if hight < img_h else img_h-1
            if bottom >= img_h:
                bottom = img_h - 1
                top = bottom - hight if bottom - hight >= 0 else 0
            if left < 0:
                left = 0
                right = width if width < img_w else img_w - 1
            if right >= img_w:
                right = img_w - 1
                left = right - width if right - width >= 0 else 0

            hight = bottom - top
            width = right - left

            #make squer
            bottom = top + min(hight, width, img_h, img_w)
            right = left + min(hight, width, img_h, img_w)

            hight = bottom - top
            width = right - left

            # move the up the face square
            move_up_by_ratio_pix = int(hight * move_up_by_ratio)
            if move_up_by_ratio_pix > 0:
                move_up_by_ratio_pix = min(move_up_by_ratio_pix, top)
            else:
                move_up_by_ratio_pix = max(move_up_by_ratio_pix, bottom - img_h) + 1
            top -= move_up_by_ratio_pix
            bottom -= move_up_by_ratio_pix

            # last checks
            if top < 0:
                top = 0
            if bottom >= img_h:
                bottom = img_h - 1
            if left < 0:
                left = 0
            if right >= img_w:
                right = img_w - 1
        curr_face_image = img[top:bottom, left:right]
        curr_face_image = transform(Image.fromarray(curr_face_image)).unsqueeze(0).cuda()
        content, _ = encode_s[0](curr_face_image)
        res_img = decode_s[0](content, entropy, curr_face_image).detach().cpu().squeeze(0)
        res_img = transforms.Normalize(mean=(-1 * mean / std).tolist(), std=(1.0 / std).tolist())(res_img)
        # resize the network output to fit the original image
        transforms_size_prossesing = [transforms.ToPILImage(), transforms.Resize(size=(hight, width)), transforms.ToTensor()]
        transforms_size_prossesing = transforms.Compose(transforms_size_prossesing)
        res_img = transforms_size_prossesing(res_img)
        if bottom - top < res_img.shape[1]:
            bottom += 1
        if right - left < res_img.shape[2]:
            left += 1
        final_res_img[:, top:bottom, left:right] = transforms_size_prossesing(res_img)

        curr_face_image = curr_face_image.cpu().squeeze(0)
        curr_face_image = transforms.Normalize(mean=(-1 * mean / std).tolist(), std=(1.0 / std).tolist())(curr_face_image)
        in_img[:, top:bottom, left:right] = transforms_size_prossesing(curr_face_image.cpu().squeeze(0))
    if use_face_locations:
        if do_pad_with_zeros_if_not_squared:
            if up_index > 0:
                final_res_img = final_res_img[:, up_index:-up_index, :]
                in_img = in_img[:, up_index:-up_index, :]
            if left_index > 0:
                final_res_img = final_res_img[:, :, left_index:-left_index]
                in_img = in_img[:, :, left_index:-left_index]

    return in_img, final_res_img
    #save_image(final_res_img, out_im_path)
    #save_image(in_img, in_im_path)
    #return in_im_path, out_im_path






@runway.setup(options={'checkpoint': runway.file(is_directory=True)})
def setup(opts):
    pass # return ct.setup_cartoonize()
    
@runway.command('translate', inputs={'image': runway.image}, outputs={'image': runway.image})
def translate(net, inputs):
    print("Starting")
    #output = ct.cartoonize(inputs['image'], "test_code/saved_models", net)
    config = get_config(selected_config)
    #input_dim = config['input_dim_a'] if opts.a2b else config['input_dim_b']
    #council_size = config['council']['council_size']

    # Setup model and data loader
    #if not 'new_size_a' in config.keys():
        #config['new_size_a'] = config['new_size']
    #is_data_A = opts.a2b

    style_dim = config['gen']['style_dim']
    h = 256
    w =  256
    max_added_val = 50
    slidermax = 6.5
    slidermin = 0.5
    sliderval = 0.27 * (slidermax - slidermin) + slidermin
    random_entropy = Variable(torch.randn(1, style_dim, 1, 1).cuda())
    random_entropy_direction = Variable(torch.randn(1, style_dim, 1, 1).cuda())
    random_entropy_direction /= torch.norm(random_entropy_direction)
    random_entropy_direction_mult = (sliderval - slidermax / 2) / (slidermax)
    random_entropy = random_entropy + max_added_val * random_entropy_direction * random_entropy_direction_mult
            
    in_im, res_im = run_net_work(
        img=inputs['image'], entropy=random_entropy,
        # use_face_locations=use_face_locations,
        # face_increes_by_dev_ratio=face_increes_by_dev_ratio, 
    )
            
    print("Done")
    return Image.fromarray(res_im)

if __name__ == '__main__':
    runway.run(port=8889)