import pdfplumber
from pdfplumber.utils import extract_text, get_bbox_overlap, obj_to_bbox
import pytesseract
import cv2
import numpy as np 
import os 
import pandas as pd
class PDFService:
      def __init__(self,dir_path):
          self.dir=dir_path
            
      def retrieve_pdf_elements(self,doc_path):
            data=[]
            with pdfplumber.open(doc_path) as pdf:
                for id,page in enumerate(pdf.pages):
                    page_height=page.height
                    filtered_page = page
                    chars=page.chars
                    for table in page.find_tables():
                        first_table_char = page.crop(table.bbox).chars[0]
                        filtered_page = filtered_page.filter(lambda obj: 
                            get_bbox_overlap(obj_to_bbox(obj), table.bbox) is None
                        )
                        chars = filtered_page.chars
                        df = pd.DataFrame(table.extract())
                        df.columns = df.iloc[0]
                        markdown = df.drop(0).to_markdown(index=False)
                        chars.append(first_table_char | {"text": markdown})
                    page_text = extract_text(chars, layout=True)
                    images=page.images
                    imgs_to_text=[]
                    for img in images:
                        image_obj = self.crop_image_from_page(page, page_height, img)
                        pil_image = image_obj.original
                        txt_image = self.extract_text_from_image(pil_image)
                        imgs_to_text.append(txt_image)
        
                    data.append({
                    "page_id":id,
                    "text":page_text,
                    
                    "images_to_text":"\n".join(imgs_to_text)
                    })
                    if id== 4:
                        break
            
            return data

      def extract_text_from_image(self,pil_image):
            open_cv_image = np.array(pil_image)
            open_cv_image = open_cv_image[:, :, ::-1].copy()
            gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
            _, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
            rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))

            dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1)

            contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, 
                                                                        cv2.CHAIN_APPROX_NONE)

            im2 = open_cv_image.copy()
            txt_image=""
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)                    
                cropped = im2[y:y + h, x:x + w]
                txt_image += pytesseract.image_to_string(cropped)
            return txt_image
      def extract_data_from_pdf_directory(self):
            """
                Retrieves all documents from a directory of PDF files.

                Args:
                    dir (str): The directory containing the PDF files.

                Returns:
                    list: A list of dictionaries, where each dictionary represents a record
                        extracted from the PDF files. Each record contains the 'text',
                        'table_to_text', and 'images_to_text' data, as well as a combined
                        'meta' field.
            """
            data=[]
            for file in os.listdir(self.dir):
                if file.endswith(".pdf"):
                    doc_path=os.path.join(self.dir,file)
                    d=self.retrieve_pdf_elements(doc_path)
                    data.extend(d)
            df = pd.DataFrame(data=data)
            df['meta']={
                "source":self.dir
            }
            records=df.to_dict('records')
            return records

      def crop_image_from_page(self,page, page_height, img):
            x0, y0, x1, y1 = img['x0'], page_height - img['y1'], img['x1'], page_height - img['y0']
            x0 = max(0, x0)
            y0 = max(0, y0)
            x1 = min(page.width, x1)
            y1 = min(page.height, y1)
            bbox = (x0, y0, x1, y1)
            cropped_page = page.crop(bbox)
            image_obj = cropped_page.to_image(resolution=400)
            return image_obj

            

