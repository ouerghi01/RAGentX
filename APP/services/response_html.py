def generate_answer_html(question, final_answer,bool):
    if bool:
        html_user= f"""
        <div class="message user" style="display: flex; flex-direction: row;">
                            <div class="message-icon">
                                <img src="/static/icons8-user.svg" alt="bot" class="bot-icon">
                            </div>
                            <div class="message-content">
                                <p style="color:rgb(50, 47, 47); font-size: 15px; line-height: 1.5; margin: 0; padding: 10px;  border-radius: 10px;">{question}</p>
        </div>
        </div>

        """
        message_html = html_user + f"""
            <div class="message ai" style="display: flex;  ">
                    <div class="message-icon">
                    <img src="/static/bot.png" alt="bot" class="bot-icon">
                    </div>
                    <div id="ai_answer" class="message-content">
                    {final_answer}
                   <p class="error-para"></p>
                    <button class="btn" id="convertBtn">
                       <i class="fa-solid fa-volume-high"></i>
                    </button>
                    </div>
             </div>
            """
        return message_html
    else:
        message_html =  f"""
         
            <div class="message ai" style="display: flex;  ">
                    <div class="message-icon">
                    <img src="/static/bot.png" alt="bot" class="bot-icon">
                    </div>
                    <div id="ai_answer" class="message-content">
                    {final_answer}
                   <p class="error-para"></p>
                    <button class="btn" id="convertBtn">
                       <i class="fa-solid fa-volume-high"></i>
                    </button>
                    </div>
             </div>
            """
        return message_html