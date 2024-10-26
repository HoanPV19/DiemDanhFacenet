import os
import shutil
from tkinter import Tk, messagebox, filedialog


def copy_excel_to_C():
    # Hiển thị cửa sổ chọn file Excel
    root = Tk()
    root.withdraw()  # Ẩn cửa sổ chính của Tkinter
    file_path = filedialog.askopenfilename(title="Chọn tệp file Excel Danh Sách Lớp", filetypes=[("Excel files", "*.xlsx *.xls")])
    
    if not file_path:
        messagebox.showinfo("Thông báo", "Không có tệp nào được chọn. Thoát chương trình.")
        return
    
    # Thư mục C cố định (bạn có thể điều chỉnh đường dẫn này)
    folder_C = 'Dataset/DSLOP'  # Thay bằng đường dẫn cố định của bạn
    
    # Tạo folder C nếu chưa có
    os.makedirs(folder_C, exist_ok=True)
    
    # Lấy tên file từ đường dẫn đã chọn
    file_name = os.path.basename(file_path)
    destination_path = os.path.join(folder_C, file_name)
    
    # Kiểm tra xem file đã tồn tại trong folder C chưa
    if os.path.exists(destination_path):
        messagebox.showinfo("Thông báo", "Danh Sách Lớp đã tồn tại.")
    else:
        try:
            # Sao chép tệp Excel vào folder C
            shutil.copy(file_path, folder_C)
            messagebox.showinfo("Thông báo", "Thêm Danh Sách Lớp Thành Công")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Có lỗi xảy ra: {e}")


# Gọi hàm
copy_excel_to_C()
