from flask import Blueprint, render_template, request, redirect, url_for, jsonify
import os
import fitz

# Create a Blueprint named 'pdf2img'
pdf2img_bp = Blueprint('pdf2img', __name__, template_folder='templates')

@pdf2img_bp.route('/pdf2img')
def show_pdf2img():
    pdf_files = [f for f in os.listdir('uploads/pdf/') if f.lower().endswith('.pdf')]  # Filter only PDF files
    return render_template('pdf2img.html', pdf_files=pdf_files)

@pdf2img_bp.route('/convert_to_img', methods=['POST'])
def convert_to_img():
    selected_files = request.form.getlist('selected_pdfs')
    selected_dpi = int(request.form['dpi'])
    try:
        convert_pdfs_to_images(selected_files, selected_dpi)
        return jsonify({"status": "success", "message": "Files converted successfully"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


def convert_pdfs_to_images(filenames, dpi):
    folderpath = 'uploads/pdf/'
    imgpath = 'uploads/images/'
    os.makedirs(imgpath, exist_ok=True)  # Ensure the target directory exists

    for filename in filenames:
        pdffile = os.path.join(folderpath, filename)
        if pdffile.lower().endswith('.pdf'):
            doc = fitz.open(pdffile)
            for page in doc:
                pix = page.get_pixmap(dpi=dpi)
                output_file = f"{imgpath}{filename[:-4]}_page{page.number}.jpg"
                pix.save(output_file)
            doc.close()
        else:
            print(f'File name: {filename} is not valid!')