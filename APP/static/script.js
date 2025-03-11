


function showUIUploadPDF() {
  const uploadpdf = document.getElementById('uploadpdf');
  uploadpdf.innerHTML = ""; // Clear previous content
  
  const title = document.createElement("h3");
  title.textContent = "Upload a document for additional context";
  title.style.textAlign = "center";
  title.style.color = "white";
  title.style.fontWeight = "bold";

  // i want title to be bold
  
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

function showLoggedInState(username) {
  document.getElementById('messanger').style.display = 'flex';
  document.getElementById('login-tab').style.display = 'none';
  document.getElementById('register-tab').style.display = 'none';
  document.getElementById('login').style.display = 'none';
  document.getElementById('logout-tab').style.display = 'flex';
  const p_new = document.createElement('p');
  p_new.textContent = " Welcome logged in as " + username;
  p_new.style.color = "white";
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
    div_new.remove();
    button_logout.remove();
  };
  document.getElementById('logout-tab').innerHTML = "";
  document.getElementById('logout-tab').appendChild(p_new);
  document.getElementById('logout-tab').appendChild(button_logout);
}

if (typeof(Storage) !== "undefined") {
    // Code for localStorage/sessionStorage.
    const jwt = localStorage.getItem('jwt');
    if(jwt){
      document.getElementById('messanger').style.display = 'flex';
      document.getElementById('login-tab').style.display = 'none';
      document.getElementById('register-tab').style.display = 'none';
      document.getElementById('login').style.display = 'none';
      document.getElementById('logout-tab').style.display = 'flex';
      const username= localStorage.getItem('username');
      showLoggedInState(username);
      showUIUploadPDF();
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
  document.addEventListener("htmx:afterRequest", function(event) {
      // Check if the request was for login
      if (event.detail.elt.closest("#login")) {
        const response =JSON.parse( event.detail.xhr.response)
          const username= response.the_user;
          console.log("Login request completed!");
          showLoggedInState(username);
        
          localStorage.setItem('jwt', response.access_token);
          localStorage.setItem('username', username);
          //alert("Login Successful!"); // Show success message
          showUIUploadPDF();
      }

      // Check if the request was for registration
      if (event.detail.elt.closest("#register")) {
          console.log("Register request completed!");
          alert("Registration Successful!");
      }
      if(event.detail.elt.closest("#form_send_message")){
        document.getElementById('messages').style.display = 'flex';
        
        console.log("Message request completed!");
      }
  });
