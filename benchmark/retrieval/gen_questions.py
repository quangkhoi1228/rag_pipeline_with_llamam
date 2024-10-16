import os
from groq import Groq
import pandas as pd
import json
import time

# Read the secret key from the file
with open("secret.txt", "r") as file:
    api_key = file.read().strip()

# Initialize the Groq client
client = Groq(api_key=api_key)


def get_answer(prompt, model="llama-3.1-70b-versatile"):
    # Example chat completion
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model=model,  # You can change this to other available models
    )

    return chat_completion.choices[0].message.content


answer_template = '''
{
    "câu hỏi dễ": {
        "câu hỏi": "...",
        "câu trả lời": """..."""
    },
    "câu hỏi trung bình": {
        "câu hỏi": "...",
        "câu trả lời": """..."""
    },
    "câu hỏi khó": {
        "câu hỏi": "...",
        "câu trả lời": """..."""
    }
}
'''

answer_sample = '''
{
    "câu hỏi": "Mức xử phạt đối với hành vi cản trở, gây khó khăn cho việc sử dụng đất của người khác là bao nhiêu tiền?",
    "câu trả lời": """
        Căn cứ theo Điều 15 Nghị định 123/2024/NĐ-CP quy định về cản trở, gây khó khăn cho việc sử dụng đất của người khác như sau:

            `Điều 15. Cản trở, gây khó khăn cho việc sử dụng đất của người khác

            1. Phạt tiền từ 1.000.000 đồng đến 3.000.000 đồng đối với hành vi đưa vật liệu xây dựng hoặc các vật khác lên thửa đất thuộc quyền sử dụng của người khác hoặc thửa đất thuộc quyền sử dụng của mình mà cản trở, gây khó khăn cho việc sử dụng đất của người khác.

            2. Phạt tiền từ 5.000.000 đồng đến 10.000.000 đồng đối với hành vi đào bới, xây tường, làm hàng rào trên đất thuộc quyền sử dụng của mình hoặc của người khác mà cản trở, gây khó khăn cho việc sử dụng đất của người khác.

            3. Biện pháp khắc phục hậu quả:

            Buộc khôi phục lại tình trạng ban đầu của đất trước khi vi phạm.

        Căn cứ tại khoản 2 Điều 5 Nghị định 123/2024/NĐ-CP quy định về mức phạt tiền như sau:

            Điều 5. Mức phạt tiền và thẩm quyền xử phạt

            [...]

            2. Mức phạt tiền quy định tại Chương II của Nghị định này áp dụng đối với cá nhân (trừ khoản 4, 5, 6 Điều 18, khoản 1 Điều 19, điểm b khoản 1 và khoản 4 Điều 20, Điều 22, khoản 2 và khoản 3 Điều 29 Nghị định này). Mức phạt tiền đối với tổ chức bằng 02 lần mức phạt tiền đối với cá nhân có cùng một hành vi vi phạm hành chính.

            [...]`

        Như vậy, mức xử phạt đối với hành vi cản trở, gây khó khăn cho việc sử dụng đất của người khác cụ thể là:

        - Phạt tiền từ 1.000.000 đồng đến 3.000.000 đồng đối với hành vi đưa vật liệu xây dựng hoặc các vật khác lên thửa đất thuộc quyền sử dụng của người khác hoặc thửa đất thuộc quyền sử dụng của mình mà cản trở, gây khó khăn cho việc sử dụng đất của người khác.

        - Phạt tiền từ 5.000.000 đồng đến 10.000.000 đồng đối với hành vi đào bới, xây tường, làm hàng rào trên đất thuộc quyền sử dụng của mình hoặc của người khác mà cản trở, gây khó khăn cho việc sử dụng đất của người khác.

        - Đồng thời, biện pháp khắc phục hậu quả là buộc khôi phục lại tình trạng ban đầu của đất trước khi vi phạm.

        Lưu ý: Mức phạt tiền quy định trên áp dụng đối với cá nhân. Mức phạt tiền đối với tổ chức bằng 02 lần mức phạt tiền đối với cá nhân có cùng một hành vi vi phạm hành chính.
    """
}, 
{
    "câu hỏi": "Các phương pháp định giá đất bao gồm những phương pháp nào?",
    "câu trả lời": """
        Căn cứ tại khoản 5 Điều 158 Luật Đất đai 2024 quy định về phương pháp định giá đất như sau:

        (1) Phương pháp so sánh được thực hiện bằng cách điều chỉnh mức giá của các thửa đất có cùng mục đích sử dụng đất, tương đồng nhất định về các yếu tố có ảnh hưởng đến giá đất đã chuyển nhượng trên thị trường, đã trúng đấu giá quyền sử dụng đất mà người trúng đấu giá đã hoàn thành nghĩa vụ tài chính theo quyết định trúng đấu giá thông qua việc phân tích, so sánh các yếu tố ảnh hưởng đến giá đất sau khi đã loại trừ giá trị tài sản gắn liền với đất (nếu có) để xác định giá của thửa đất cần định giá;

        (2) Phương pháp thu nhập được thực hiện bằng cách lấy thu nhập ròng bình quân năm trên một diện tích đất chia cho lãi suất tiền gửi tiết kiệm bình quân của loại tiền gửi bằng tiền Việt Nam kỳ hạn 12 tháng tại các ngân hàng thương mại do Nhà nước nắm giữ trên 50% vốn điều lệ hoặc tổng số cổ phần có quyền biểu quyết trên địa bàn cấp tỉnh của 03 năm liền kề tính đến hết quý gần nhất có số liệu trước thời điểm định giá đất;

        (3) Phương pháp thặng dư được thực hiện bằng cách lấy tổng doanh thu phát triển ước tính trừ đi tổng chi phí phát triển ước tính của thửa đất, khu đất trên cơ sở sử dụng đất có hiệu quả cao nhất (hệ số sử dụng đất, mật độ xây dựng, số tầng cao tối đa của công trình) theo quy hoạch sử dụng đất, quy hoạch chi tiết xây dựng đã được cơ quan có thẩm quyền phê duyệt;

        (4) Phương pháp hệ số điều chỉnh giá đất được thực hiện bằng cách lấy giá đất trong bảng giá đất nhân với hệ số điều chỉnh giá đất. Hệ số điều chỉnh giá đất được xác định thông qua việc so sánh giá đất trong bảng giá đất với giá đất thị trường;

        (5) Chính phủ quy định phương pháp định giá đất khác chưa được quy định tại (1), (2), (3) và (4) sau khi được sự đồng ý của Ủy ban Thường vụ Quốc hội.
    """
}
'''

prompt_template = """
Bạn đang đóng vai là một luật sư rất giỏi chuyên về luật bất động sản.

Bạn được cung cấp một phần văn bản pháp luật.
Hãy tưởng tượng câu hỏi nào của khách hàng bạn cần dùng nội dung của văn bản pháp luật đã cung cấp để trả lời? 
Hãy cho tôi 3 câu hỏi (1 câu khó, 1 câu trung bình, 1 câu dễ) mà bạn cần dùng nội dung của văn bản pháp luật đã cung cấp để trả lời. 
Với mỗi câu hỏi, hãy cung cấp câu trả lời cho câu hỏi đó. 

Câu hỏi nên tập trung đặt vấn đề về bất động sản. Không được hỏi về các văn bản pháp luật hoặc hỏi cụ thể về nội dung của văn bản pháp luật. Câu hỏi không được bao gồm tên của văn bản pháp luật, tên viết tắt của văn bản pháp luật hay số hiệu của nó.
Trong câu trả lời, bạn phải nói rõ phần nào của văn bản pháp luật bạn đã lấy thông tin. Khi nói đến tên của văn bản, bạn không được sử dụng từ thay thế cho tên của văn bản, bạn không được sử dụng từ `Quyết định này` hoặc các từ tương tự để thay thế tên của văn bản.

Dưới đây là 2 ví dụ về câu hỏi từ khách hàng và cách bạn trả lời: {answer_sample}

Trả lời bằng tiếng Việt. 
Trả lời dạng python dictionary với định dạng: 
{answer_template}

Tên văn bản pháp luật: {document_name}
Văn bản pháp luật: {document}
"""


# prompt_template = """
# You are a lawyer who is very good at providing legal advice on real estate.
# You are provided with a part of the legal document.
# Here are 2 examples of questions from client and how you answer them: {answer_sample}
# Imagine which questions of the client do you need to use the content of the legal document provided to answer?
# Give me 3 questions (1 difficult question, 1 medium question, 1 easy question) that you need to use the content in the legal document provided to answer.
# For each question, provide your answer to the question.
# The question should not include the name of the legal document or its abbreviation.
# In the answer, you must mention which part of the legal document you got the information from. When mentioning the name of the document, you need to specify the name of the document, you cannot use the word to replace the name of the document, you cannot use the word `Quyết định này` or similar words to replace the name of the document.
# Answer in Vietnamese.
# Answer in JSON format with the format:
# {answer_template}

# The name of the legal document: {document_name}
# The legal document: {document}
# """

df = pd.read_parquet("../300_samples.parquet")

for index, row in df.iterrows():
    if index <= 240:
        continue
    
    id = row["id"]
    relevant_chunks = len(row["content_text"])
    document_name = row["title"]
    print(index)
    document = row["content_text_raw"].strip()[:5000]
    prompt = prompt_template.format(
        document_name=document_name,
        document=document,
        answer_template=answer_template,
        answer_sample=answer_sample,
    )
    answer = get_answer(prompt)
    with open(f"benchmark/raw.txt", "a", encoding="utf-8") as file:
        file.write("id: " + str(id) + "\n")
        file.write("document_name: " + document_name + "\n")
        file.write("relevant_chunks: " + str(relevant_chunks) + "\n")
        file.write(answer + "\n")
        file.write("-" * 100 + "\n")
    # answer_json = eval(answer)
    # answer_json["id"] = id
    # answer_json["document_name"] = document_name
    # answer_json["relevant_chunks"] = relevant_chunks
    # with open(f"benchmark/eval.jsonl", "a", encoding="utf-8") as file:
    #     json.dump(answer_json, file)
    #     file.write("\n")
    # print(answer_json)
    time.sleep(3)
