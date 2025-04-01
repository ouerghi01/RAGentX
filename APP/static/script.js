
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
    title.style.color = "white";
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
  title.textContent = "Upload a document for additional context";
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
  promptText.textContent = "Drag and drop a file here or click to upload";
  promptText.style.color = "#555";
  dropArea.appendChild(promptText);
  
  const fileLimit = document.createElement("p");
  fileLimit.textContent = "Limit: 200 MB per file";
  fileLimit.style.color = "#777";
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

function showLoggedInState(username,job) {
  document.getElementById('messanger').style.display = 'flex';
  document.getElementById('login-tab').style.display = 'none';
  document.getElementById('register-tab').style.display = 'none';
  document.getElementById('login').style.display = 'none';
  document.getElementById('logout-tab').style.display = 'flex';
  const p_new = document.createElement('p');
  p_new.textContent = " Welcome logged in as " + username 
  p_new.style.color = "white";
  const p_nn = document.createElement('p');
  p_nn.textContent = "Your job is " + job;
  p_nn.style.color = "green";
  const button_logout = document.createElement('button');
  button_logout.innerHTML = "Logout";
  button_logout.style.width = "fit-content";
  button_logout.style.padding = "8px 15px";
  button_logout.style.border = "none";
  button_logout.style.borderRadius = "5px";
  button_logout.style.backgroundColor = "#007bff";
  button_logout.style.color = "#EAECEF";
  button_logout.style.cursor = "pointer";
  button_logout.style.marginTop = "10px";
  button_logout.style.fontWeight = "bold";
  button_logout.style.transition = "background-color 0.3s";
  button_logout.style.position = "relative";
  button_logout.style.right = "15%";
  button_logout.style.alignItems = "center";
  button_logout.style.transform = "translateX(30%)";
  button_logout.style.justifyContent = "center";
  button_logout.style.display = "flex";

  button_logout.onmouseover = function () { this.style.backgroundColor = "#0056b3"; };
  button_logout.onmouseout = function () { this.style.backgroundColor = "#007bff"; };


  button_logout.onclick = function () {
    localStorage.removeItem('jwt');
    document.getElementById('messanger').style.display = 'none';
    document.getElementById('login-tab').style.display = 'block';
    document.getElementById('register-tab').style.display = 'block';
    document.getElementById('login').style.display = 'block';
    document.getElementById('logout-tab').style.display = 'none';
    document.getElementById('uploadpdf').style.display='none';
    document.getElementById('questions').style.display='none';
    button_logout.remove();
      
  };
  document.getElementById('logout-tab').innerHTML = "";
  document.getElementById('logout-tab').appendChild(p_new);
    document.getElementById('logout-tab').appendChild(p_nn);
  document.getElementById('logout-tab').appendChild(button_logout);
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
    console.log(verified_jwt);
    if(verified_jwt && jwt){
      document.getElementById('messanger').style.display = 'flex';
      document.getElementById('login-tab').style.display = 'none';
      document.getElementById('register-tab').style.display = 'none';
      document.getElementById('login').style.display = 'none';
      document.getElementById('logout-tab').style.display = 'flex';
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
    document.addEventListener('DOMContentLoaded', () => {
        const input = document.getElementById('input');
        input.addEventListener('input', () => {
          const button = input.nextElementSibling;
          button.disabled = !input.value;
        });
      });
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
  document.addEventListener("htmx:afterRequest", async function(event) {
      // Check if the request was for login
      if (event.detail.elt.closest("#login")) {
        const response =JSON.parse( event.detail.xhr.response)
        
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
          alert("Registration Successful!");
      }
      
      if(event.detail.elt.closest("#form_send_message")){
        
        document.getElementById('messages').style.display = 'flex';
        messages_ai = document.getElementsByClassName('message ai');
        for (let i = 0; i < messages_ai.length; i++) {
            const element = messages_ai[i]
            let ai_answer = element.querySelector("#ai_answer");
            if (ai_answer) {
                const convertBtn = element.querySelector("#convertBtn");

                let enteredText = removeTags(ai_answer.textContent);
               
                const apiKey="AIzaSyD59T_Pyw4rzPrU90s_64Ctp2kOWBfKH9Q"
                const userInput = `Explain the content of this text while ignoring any HTML or CSS: ${enteredText}`;
                const res = await fetch("https://generativelanguage.googleapis.com/v1/models/gemini-1.5-pro:generateContent?key=" + apiKey, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ contents: [{ parts: [{ text: userInput }] }] })
                });
                const data = await res.json();
                enteredText = data.candidates?.[0]?.content?.parts?.[0]?.text || "No response";
                convertBtn.innerHTML = '<i class="fa-solid fa-volume-high"></i>';
                let isSpeaking = false; // Track speaking state
                convertBtn.addEventListener('click', function () {
                const speechSynth = window.speechSynthesis;
                       
                const error = element.querySelector('.error-para');
                console.log(enteredText)
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
