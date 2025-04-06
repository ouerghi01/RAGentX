
function showAlert(message) {
    const alertBox = document.getElementById("error-alert");
    alertBox.style.display="flex";
    const alertText = document.getElementById("alert-text");
  
    alertText.textContent = message;
    alertBox.classList.remove("hidden");
  
    setTimeout(() => {
      closeAlert();
    }, 5000); // Auto-close after 5 seconds
  }
  
  function closeAlert() {
    document.getElementById("error-alert").style.display='none';
  }
  function showSuccessAlert(message) {
    const alertBox = document.getElementById("success-alert");
    alertBox.style.display="flex";
    const alertText = document.getElementById("success-alert-text");
  
    alertText.textContent = message;
    alertBox.classList.remove("hidden");
  
    setTimeout(() => {
      closeSuccessAlert();
    }, 5000); // Auto-close after 5 seconds
  }
  
  function closeSuccessAlert() {
    document.getElementById("success-alert").style.display='none';
  }
function tryFollowingQuestion() {
    const questions = document.getElementById('questions');
    questions.style.display = "flex";
    questions.innerHTML = ""; // Clear previous content
    
    const questionsExp = [
        "Tell me who you assist as an agent.",
        "What is the main idea of the text?",
       
    ];
    
    const title = document.createElement('h4');
    title.textContent = "Try asking these questions:";
    title.style.color = "#ffffff";
    title.style.marginBottom = "10px";
    title.style.textAlign = "center";
    questions.appendChild(title);
    
    for (let i = 0; i < questionsExp.length; i++) {
        const question = document.createElement('div');
        question.classList.add('question');
        question.textContent = questionsExp[i];
        
        // Style the question elements
        question.style.padding = "8px 12px";
        question.style.margin = "5px 0";
        question.style.backgroundColor = "#2e3642";
        question.style.color = "#e6e6e6";
        question.style.borderRadius = "5px";
        question.style.cursor = "pointer";
        question.style.transition = "all 0.3s ease";
        question.style.border = "1px solid #3e4758";
        
        // Add hover effects
        question.onmouseover = function() {
            this.style.backgroundColor = "#3e4758";
            this.style.transform = "translateY(-2px)";
        };
        question.onmouseout = function() {
            this.style.backgroundColor = "#2e3642";
            this.style.transform = "translateY(0)";
        };
        
        // Add click functionality to use the question
        question.onclick = function() {
            const input = document.getElementById('input');
            if (input) {
                input.value = this.textContent;
                input.focus();
                // Enable the send button if it exists
                const button = input.nextElementSibling;
                if (button) button.disabled = false;
            }
        };
        
        questions.appendChild(question);
    }
}
function showUIUploadPDF() {
  const uploadpdf = document.getElementById('uploadpdf');
  uploadpdf.innerHTML = ""; // Clear previous content
  
  const title = document.createElement("h3");
  title.textContent = "Upload context file";
  title.style.textAlign = "center";
  title.style.color = "white";
  title.style.fontWeight = "bold";
  
  uploadpdf.appendChild(title);
  
  const dropArea = document.createElement("div");
  dropArea.classList.add("upload-area");
  dropArea.style.display = "flex";
  dropArea.style.flexDirection = "column";
  dropArea.style.justifyContent = "center";
  dropArea.style.alignItems = "center";
  dropArea.style.height = "100px";
  dropArea.style.border = "2px dashed #007bff";
  dropArea.style.borderRadius = "10px";
  dropArea.style.backgroundColor = "#f8f9fa";
  dropArea.style.padding = "20px";
  dropArea.style.textAlign = "center";
  dropArea.style.cursor = "pointer";
  dropArea.style.transition = "0.3s ease-in-out";
  dropArea.addEventListener("dragover", (e) => {
      e.preventDefault();
      dropArea.style.backgroundColor = "#e3f2fd";
  });
  dropArea.addEventListener("dragleave", () => {
      dropArea.style.backgroundColor = "#f8f9fa";
  });
  dropArea.addEventListener("drop", (e) => {
      e.preventDefault();
      dropArea.style.backgroundColor = "#f8f9fa";
      const files = e.dataTransfer.files;
      if (files.length > 0) {
          input.files = files;
      }
  });
  
  const promptText = document.createElement("p");
  promptText.textContent = "Drop or click to upload";
  promptText.style.color = "#2a3439";
  dropArea.appendChild(promptText);
  
  const fileLimit = document.createElement("p");
  fileLimit.textContent = "Limit: 200 MB per file";
  fileLimit.style.color = "#2a3439";
  fileLimit.style.fontSize = "14px";
  dropArea.appendChild(fileLimit);
  
  const input = document.createElement("input");
  input.type = "file";
  input.id = "file";
  input.name = "file";
  input.accept = ".pdf";
  input.multiple = false;
  input.style.display = "none";
  
  dropArea.addEventListener("click", () => input.click());
  input.addEventListener("change", () => {
      if (input.files.length > 0) {
          promptText.textContent = `Selected file: ${input.files[0].name}`;
          promptText.style.color = "green";
      }
  });
  
  dropArea.appendChild(input);
  uploadpdf.appendChild(dropArea);
  
  const saveButton = document.createElement("button");
  saveButton.textContent = "Upload";
  saveButton.style.marginTop = "15px";
  saveButton.style.padding = "10px 20px";
  saveButton.style.border = "none";
  saveButton.style.borderRadius = "5px";
  saveButton.style.backgroundColor = "#007bff";
  saveButton.style.color = "white";
  saveButton.style.fontSize = "16px";
  saveButton.style.cursor = "pointer";
  saveButton.style.transition = "0.3s";
  saveButton.addEventListener("mouseenter", () => saveButton.style.backgroundColor = "#0056b3");
  saveButton.addEventListener("mouseleave", () => saveButton.style.backgroundColor = "#007bff");
  uploadpdf.appendChild(saveButton);
  saveButton.onclick = function () {
      const file = input.files[0];
      if (!file) {
          alert("Please select a file to upload");
          return;
      }
      if (file.size > 200 * 1024 * 1024) {
          alert("File size exceeds the limit of 200 MB");
          return;
      }
      const formData = new FormData();
      formData.append("file", file);
      
      fetch("/uploads/", {
          method: "POST",
          body: formData,
      })
          .then((response) => response.json())
          .then((data) => {
              alert(data.message);
          })
          .catch((error) => {
              console.error("Error:", error);
          });
  }

}

function showLoggedInState(username, job) {
    const messanger = document.getElementById('messanger');
    const loginTab = document.getElementById('login-tab');
    const registerTab = document.getElementById('register-tab');
    const login = document.getElementById('login');
    const logoutTab = document.getElementById('logout-tab');
    const nav_right= document.getElementById('nav_right');

  
    // Show/hide elements
    messanger.style.display = 'flex';
    loginTab.style.display = 'none';
    registerTab.style.display = 'none';
    login.style.display = 'none';
    logoutTab.style.display = 'flex';
    nav_right.style.display = 'flex';
  
    // Clean previous content
    logoutTab.innerHTML = "";
  
    // Style the logout-tab container
    logoutTab.style.alignItems = 'center';
    logoutTab.style.gap = '15px';
    logoutTab.style.color = '#fff';
    logoutTab.style.fontSize = '16px';
    logoutTab.style.fontWeight = '500';
  
    // Info text container (username + job in same line)
    const infoText = document.createElement('span');
    infoText.innerHTML = `👋 Welcome, <strong>${username}</strong> &nbsp;&nbsp;|&nbsp;&nbsp; 💼 <span style="color:lightgreen">${job}</span>`;
    
    // Logout button
    const buttonLogout = document.createElement('button');
    buttonLogout.innerText = "Logout";
    Object.assign(buttonLogout.style, {
      padding: "6px 14px",
      border: "none",
      borderRadius: "6px",
      backgroundColor: "#dc3545",
      color: "#fff",
      cursor: "pointer",
      fontWeight: "bold",
      fontSize: "14px",
      transition: "background-color 0.3s",
    });
  
    buttonLogout.onmouseover = () => {
      buttonLogout.style.backgroundColor = "#c82333";
    };
    buttonLogout.onmouseout = () => {
      buttonLogout.style.backgroundColor = "#dc3545";
    };
  
    buttonLogout.onclick = function () {
      localStorage.removeItem('jwt');
      messanger.style.display = 'none';
      loginTab.style.display = 'block';
      registerTab.style.display = 'block';
      login.style.display = 'block';
      logoutTab.style.display = 'none';
      document.getElementById('uploadpdf').style.display = 'none';
      document.getElementById('questions').style.display = 'none';
      document.getElementById('nav_right').style.display= 'none';
      logoutTab.innerHTML = "";
      localStorage.clear()
      clearMessages();
    };
  
    logoutTab.appendChild(infoText);
    logoutTab.appendChild(buttonLogout);
  }
  
if (typeof(Storage) !== "undefined") {
    // Code for localStorage/sessionStorage.
    const jwt = localStorage.getItem('jwt');
    const formData= new FormData();
    formData.append("jwt", jwt);
    const verified_jwt = fetch("/verify/", {
        method: "POST",
        body: formData,
    })
        .then((response) => response.json())
        .then((data) => {
            return data 
        })
        .catch((error) => {
            console.error("Error:", error);
        });
    
    if(verified_jwt && jwt){
      document.getElementById('messanger').style.display = 'flex';
      document.getElementById('login-tab').style.display = 'none';
      document.getElementById('register-tab').style.display = 'none';
      document.getElementById('login').style.display = 'none';
      document.getElementById('logout-tab').style.display = 'flex';
      document.getElementById('nav_right').style.display = 'flex';
      const username= localStorage.getItem('username');
        const job = localStorage.getItem('his_job');
      showLoggedInState(username,job);
      showUIUploadPDF();
      tryFollowingQuestion();
    }else{
        document.getElementById('questions').style.display='none';
        document.getElementById('uploadpdf').style.display='none';
        document.getElementById('messanger').style.display = 'none';
        document.getElementById('logout-tab').style.display = 'none';
        clearMessages()
        const nav_right= document.getElementById('nav_right');
        nav_right.style.display='none';
    }
  }   
function switchTab(tab) {
      document.getElementById('login-tab').classList.remove('active');
      document.getElementById('register-tab').classList.remove('active');
      document.getElementById('login').classList.remove('active');
      document.getElementById('register').classList.remove('active');

      document.getElementById(tab + '-tab').classList.add('active');
      document.getElementById(tab).classList.add('active');
      if (tab === 'login') {
          document.getElementById('register').style.display = 'none';
          document.getElementById('login').style.display = 'flex';
      }else{
          document.getElementById('login').style.display = 'none';
          document.getElementById('register').style.display = 'flex';
          
      }
    }
    
    
    
    
function cacheMessages(messagesContainer) {
        localStorage.setItem("chat_messages", messagesContainer.innerHTML);
      }
    
document.addEventListener('DOMContentLoaded', async () => {
        const input = document.getElementById('input');
        input.addEventListener('input', () => {
          const button = input.nextElementSibling;
          button.disabled = !input.value;
        });
        const messagesContainer = document.getElementById("messages");
        const cached = localStorage.getItem("chat_messages");
        if (cached) {
        messagesContainer.innerHTML = cached;
        messagesContainer.style.display = "flex"; // show the messages
        messagesContainer.style.backgroundColor="rgb(53, 56, 57)"
        await generateSpeechFromAI();
        //document.getElementById("messanger").style.display = "block"; // show the messanger
        }
      });
async function generateSpeechFromAI() {
    container_ai = document.getElementsByClassName('message ai');
    console.log(container_ai);
    for (let i = 0; i < container_ai.length; i++) {
        const element = container_ai[i];
        let ai_answer = element.querySelector("#ai_answer");
        if (ai_answer) {
            const convertBtn = element.querySelector("#convertBtn");

            let enteredText = removeTags(ai_answer.textContent);

            const apiKey = "AIzaSyD59T_Pyw4rzPrU90s_64Ctp2kOWBfKH9Q";
            const userInput = `Convert the given text into natural-sounding speech : ${enteredText}`;
            const res = await fetch("https://generativelanguage.googleapis.com/v1/models/gemini-1.5-pro:generateContent?key=" + apiKey, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ contents: [{ parts: [{ text: userInput }] }] })
            });
            const data = await res.json();
            enteredText = data.candidates?.[0]?.content?.parts?.[0]?.text || "No response";
            enteredText = enteredText.replace(/\*+/g, '');
            convertBtn.innerHTML = '<i class="fa-solid fa-volume-high"></i>';
            let isSpeaking = false; // Track speaking state
            convertBtn.addEventListener('click', function () {
                const speechSynth = window.speechSynthesis;

                const error = element.querySelector('.error-para');
                console.log(enteredText);
                if (!enteredText.trim().length) {
                    error.textContent = `Nothing to Convert! Enter text in the text area.`;
                    return;
                }
                // If already speaking, stop it
                if (isSpeaking) {
                    speechSynth.cancel();
                    isSpeaking = false;
                    convertBtn.innerHTML = '<i class="fa-solid fa-volume-high"></i>'; // Reset button
                    return;
                }
                error.textContent = ""; // Clear error message



                // Wait for voices to load before speaking
                function speakText(text) {
                    const sentences = text.split(/(?<=[.!?])\s+/); // Split text into sentences
                    let index = 0;

                    function speakNextSentence() {
                        if (index < sentences.length) {
                            const newUtter = new SpeechSynthesisUtterance(sentences[index]);

                            // Select a voice (English by default)
                            const voices = speechSynth.getVoices();
                            newUtter.voice = voices.find(v => v.lang.startsWith('en')) || voices[0];

                            newUtter.rate = 1.0; // Normal speed

                            newUtter.onend = () => {
                                index++;
                                speakNextSentence(); // Speak the next part
                            };
                            speechSynth.speak(newUtter);
                        } else {
                            // Reset button when speech is fully done
                            convertBtn.innerHTML = '<i class="fa-solid fa-volume-high fa-beat"></i>';
                        }
                    }

                    // Cancel any existing speech before starting new
                    isSpeaking = true;
                    speechSynth.cancel();

                    // Change button to indicate speaking
                    convertBtn.innerHTML = '<i class="fa-solid fa-volume-high fa-beat"></i>';

                    speakNextSentence();
                }

                // Wait for voices to load before starting
                if (speechSynth.getVoices().length === 0) {
                    window.speechSynthesis.onvoiceschanged = function () {
                        speakText(enteredText);
                    };
                } else {
                    speakText(enteredText);
                }
            });
            convertBtn.innerHTML = '<i class="fa-solid fa-volume-high"></i>';
        }

    }
}

      function removeTags(str) {
        
        if ((str === null) || (str === ''))
            return false;
        else
            str = str.toString();
        
        // Regular expression to identify HTML tags in
        // the input string. Replacing the identified
        // HTML tag with a null string.
        
        str= str.replace(/(<([^>]+)>)/ig, '');
        htmlText = str.replace(/style="[^"]*"/g, '');
        // Remove CSS rules (everything inside { })
        let cleanText = htmlText.replace(/\{[^}]*\}/g, '');

        // Remove CSS selectors (lines that contain only a class or tag)
        cleanText = cleanText.replace(/^\s*\S+\s*$/gm, '');

        // Remove JavaScript-style comments
        cleanText = cleanText.replace(/\/\/.*$/gm, '');  // Remove single-line comments
        cleanText = cleanText.replace(/\/\*[\s\S]*?\*\//g, '');  // Remove multi-line comments

        // Remove extra spaces and newlines
        cleanText = cleanText.replace(/\s+/g, ' ').trim();
        // Remove CSS selectors and pseudo-elements
        cleanText = cleanText.replace(/\.[\w-]+(\s+|::?[\w-]+)?/g, '');
        // Remove specific tags like h2, h3, th, td, and @keyframes
        cleanText = cleanText.replace(/<\/?(h2|h3|th|td)[^>]*>/gi, ''); // Remove h2, h3, th, td tags
        cleanText = cleanText.replace(/@keyframes\s+[^{]*\{[^}]*\}/gi, ''); // Remove @keyframes rules
        cleanText = cleanText.replace(/@media\s*\(max-width:\s*600px\)[^{]*\{[^}]*\}/gi, '');
        // Remove all remaining HTML tags
        cleanText = cleanText.replace(/<\/?[^>]+(>|$)/g, '');
        // Remove specific tags and media queries
        cleanText = cleanText.replace(/<\/?(h2|h3|table|th|td)[^>]*>/gi, ''); // Remove h2, h3, table, th, td tags
        cleanText = cleanText.replace(/@media\s*\(max-width:\s*600px\)[^{]*\{[^}]*\}/gi, ''); // Remove @media queries
        text = cleanText.replace(/[a-zA-Z0-9\s,#.:>\[\]=~^$*()]+?\s*\{[^}]*\}/g, "");
        return text
        
        

    }
    document.body.addEventListener("htmx:afterSwap", async  function(evt) {
        const messages=document.getElementById('messages')
        if (evt.detail.target.id === "messages") {
          alert("Message sent successfully!");
          messages.style.display = "flex";
          document.getElementById("input").textContent="";
          cacheMessages(messages);
        

        }
      });
  document.addEventListener("htmx:afterRequest", async function(event) {
      // Check if the request was for login
      if (event.detail.elt.closest("#login")) {
         const response =JSON.parse( event.detail.xhr.response)
         console.log(response)
          if (response.detail == "Invalid username or password") {
              showAlert(response.detail);
              return;
          }else{
            showSuccessAlert("User successfully logged in!");
          }
          const username= response.the_user;
          const job = response.his_job;
          showLoggedInState(username,job);
          localStorage.clear();
          localStorage.setItem('jwt', response.access_token);
          localStorage.setItem('username', username);
          localStorage.setItem('his_job', job);
          //alert("Login Successful!"); // Show success message
          document.getElementById('uploadpdf').style.display='block';
          showUIUploadPDF();
          tryFollowingQuestion();

      }
         
      // Check if the request was for registration
      if (event.detail.elt.closest("#register")) {
          console.log("Register request completed!");
          //alert("Registration Successful!");
      }
      
      if(event.detail.elt.closest("#form_send_message")){
        const messages=document.getElementById('messages')
        messages.style.display = 'flex';
        messages.style.backgroundColor = '#353839';
        messages.style.padding = '10px';
        messages.style.borderRadius = '10px';
        await generateSpeechFromAI();

       
        
        

      }

      
    }
);
  var ws = new WebSocket("ws://localhost:8000/ws");
  ws.onmessage = function(event) {
    var input = $("#input"); // jQuery selector
    input.css("transition", "all 0.3s ease");

    try {
        var availableTags = event.data.split(',').map(tag => tag.trim());

        console.log(availableTags); // Debug: Ensure it's a valid list

        
        
        // Reinitialize autocomplete with new data
        input.autocomplete({
            source: availableTags
        });
        

    } catch (error) {
        console.error("Failed to parse server response:", error);
    }

    // Add background effect
    input.css("background-color", "#f0f8ff");
    setTimeout(() => input.css("background-color", ""), 300);
};

  function sendMessage(event) {
      var input = document.getElementById("input")
      
      if (input.value == "") {
          return
      }
      ws.send(input.value)
      
  }
  function clearMessages() {
    localStorage.removeItem("chat_messages");
  }
  function onClick_input(event) {
    const text_input = $("#input").val();
    const messages = document.getElementById("messages");
    messages.style.display='flex'
    messages.style.backgroundColor+'rgb(53, 56, 57)'
    const html_content = `<div class="message user" style="display: flex; flex-direction: row;">
                    <div class="message-icon">
                        <img src="/static/icons8-user.svg" alt="bot" class="bot-icon">
                    </div>
                    <div class="message-content">
                        <p style="color:rgb(50, 47, 47); font-size: 15px; line-height: 1.5; margin: 0; padding: 10px;  border-radius: 10px;">${text_input}</p>
                    </div>
                </div>`;
    messages.innerHTML += html_content;
    
  }