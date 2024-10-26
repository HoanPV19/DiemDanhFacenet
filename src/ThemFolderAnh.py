import os
import shutil
from tkinter import Tk, filedialog, messagebox, Toplevel
import tensorflow as tf
import facenet
import pickle
from sklearn.svm import SVC
import numpy as np
import math
import imageio
from PIL import Image
import align.detect_face


def copy_and_move_files():
    # Hiển thị cửa sổ chọn folder A
    root = Tk()
    root.withdraw()  # Ẩn cửa sổ chính

    # Hiển thị cửa sổ chọn thư mục
    folder_A = filedialog.askdirectory(title="Chọn thư mục ảnh của các sinh viên")
    root.destroy()  # Đóng cửa sổ sau khi chọn thư mục

    if not folder_A:
        messagebox.showerror("Lỗi", "Không có thư mục nào được chọn.")
        return
    
    # Lấy tên thư mục A (chỉ tên, không có đường dẫn)
    folder_A_name = os.path.basename(folder_A)
    
    # Tạo đường dẫn thư mục B với tên giống tên thư mục A
    folder_B = folder_A_name
    
    # Thư mục C cố định
    folder_C = 'Dataset/FaceData'  # Thay bằng đường dẫn cố định của bạn
    
    # Kiểm tra nếu lớp đã tồn tại trong thư mục C
    if os.path.exists(os.path.join(folder_C, folder_B)):
        messagebox.showinfo("Thông báo", f'Lớp {folder_B} đã tồn tại.')
        return  # Dừng hàm nếu lớp đã tồn tại

    # Tạo folder B với 2 folder con raw và processed bên trong folder C
    raw_folder = os.path.join(folder_C, folder_B, 'raw')
    processed_folder = os.path.join(folder_C, folder_B, 'processed')
    
    os.makedirs(processed_folder, exist_ok=True)
    
    # Xóa nếu raw_folder đã tồn tại
    if os.path.exists(raw_folder):
        messagebox.showinfo("Thông báo", f'Xóa thư mục {raw_folder} trước khi sao chép.')
        shutil.rmtree(raw_folder)  # Xóa thư mục raw đã tồn tại

    # Sao chép tất cả nội dung từ folder A vào raw_folder
    try:
        shutil.copytree(folder_A, raw_folder)
    except Exception as e:
        messagebox.showerror("Lỗi", f"Lỗi khi sao chép: {e}")
        return
    
    # Thông báo sau khi thành công
    messagebox.showinfo("Thông báo", f"Thêm {folder_B} thành công vào trong {folder_C} vui lòng chọn OK để quá trình mã hóa diễn ra...")
    
    # Mã hóa folder processed
    process_faces(os.path.join(folder_C, folder_B))  # Thực hiện xử lý
    encode_faces(os.path.join(folder_C, folder_B))  # Sau đó mã hóa


def process_faces(base_dir):
    raw_dir = os.path.join(base_dir, 'raw')
    processed_dir = os.path.join(base_dir, 'processed')

    # Tạo thư mục đầu ra nếu không tồn tại
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    print('Creating networks and loading parameters...')
    with tf.Graph().as_default():
        with tf.compat.v1.Session() as sess:
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

            minsize = 20  # kích thước tối thiểu của khuôn mặt
            threshold = [0.6, 0.7, 0.7]  # ngưỡng cho 3 bước
            factor = 0.709  # hệ số tỉ lệ

            # Lấy dataset (các hình ảnh trong raw_dir)
            dataset = facenet.get_dataset(raw_dir)
            nrof_images_total = 0
            nrof_successfully_aligned = 0

            for cls in dataset:
                output_class_dir = os.path.join(processed_dir, cls.name)
                if not os.path.exists(output_class_dir):
                    os.makedirs(output_class_dir)

                for image_path in cls.image_paths:
                    nrof_images_total += 1
                    filename = os.path.splitext(os.path.split(image_path)[1])[0]
                    output_filename = os.path.join(output_class_dir, filename + '.png')

                    if not os.path.exists(output_filename):
                        try:
                            img = imageio.imread(image_path)
                        except (IOError, ValueError, IndexError) as e:
                            print(f'Error reading {image_path}: {e}')
                            continue
                        if img.ndim < 2:
                            print(f'Unable to align {image_path}')
                            continue
                        if img.ndim == 2:
                            img = facenet.to_rgb(img)
                        img = img[:,:, 0:3]

                        bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
                        nrof_faces = bounding_boxes.shape[0]
                        if nrof_faces > 0:
                            det = bounding_boxes[:, 0:4]
                            det_arr = []
                            img_size = np.asarray(img.shape)[0:2]
                            if nrof_faces > 1:
                                bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
                                img_center = img_size / 2
                                offsets = np.vstack([(det[:, 0] + det[:, 2]) / 2 - img_center[1],
                                                     (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
                                offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
                                index = np.argmax(bounding_box_size - offset_dist_squared * 2.0)
                                det_arr.append(det[index])
                            else:
                                det_arr.append(np.squeeze(det))

                            for i, det in enumerate(det_arr):
                                det = np.squeeze(det)
                                bb = np.zeros(4, dtype=np.int32)
                                bb[0] = np.maximum(det[0], 0)
                                bb[1] = np.maximum(det[1], 0)
                                bb[2] = np.minimum(det[2], img_size[1])
                                bb[3] = np.minimum(det[3], img_size[0])
                                cropped = img[bb[1]:bb[3], bb[0]:bb[2],:]
                                cropped = Image.fromarray(cropped)
                                scaled = cropped.resize((160, 160), Image.BILINEAR)  # Resize to fixed size
                                nrof_successfully_aligned += 1
                                output_filename_n = "{}{}".format(os.path.splitext(output_filename)[0], '.png')
                                imageio.imwrite(output_filename_n, scaled)
                        else:
                            print(f'Unable to align {image_path}')

    print(f'Total number of images: {nrof_images_total}')
    print(f'Number of successfully aligned images: {nrof_successfully_aligned}')


def encode_faces(base_dir):
    print('Loading aligned faces for encoding...')
    dataset = facenet.get_dataset(os.path.join(base_dir, 'processed'))
    paths, labels = facenet.get_image_paths_and_labels(dataset)

    print(f'Number of classes: {len(dataset)}')
    print(f'Number of images: {len(paths)}')

    with tf.Graph().as_default():
        with tf.compat.v1.Session() as sess:
            # Load the model inside the graph context
            print('Loading feature extraction model...')
            facenet.load_model('Models/20180402-114759.pb')  # Đường dẫn tới model

            images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            print('Calculating features for images...')
            nrof_images = len(paths)
            nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / 100))
            emb_array = np.zeros((nrof_images, embedding_size))

            for i in range(nrof_batches_per_epoch):
                start_index = i * 100
                end_index = min((i + 1) * 100, nrof_images)
                paths_batch = paths[start_index:end_index]
                images = facenet.load_data(paths_batch, False, False, 160)
                feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)

            # Train classifier
            classifier_filename_exp = os.path.join('Models/vector', os.path.basename(base_dir) + '.pkl')
            print('Training classifier...')
            model = SVC(kernel='linear', probability=True)
            model.fit(emb_array, labels)

            # Create a list of class names
            class_names = [cls.name.replace('_', ' ') for cls in dataset]

            # Saving classifier model
            with open(classifier_filename_exp, 'wb') as outfile:
                pickle.dump((model, class_names), outfile)
            print(f'Saved classifier model to file "{classifier_filename_exp}"')

    # Hiển thị thông báo khi hoàn thành mã hóa
    messagebox.showinfo("Thông báo", "Mã hóa khuôn mặt đã hoàn thành!")


if __name__ == "__main__":
    copy_and_move_files()
