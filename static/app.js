document.getElementById("uploadBtnA").onchange = function () {
    document.getElementById("uploadFileA").value = this.files[0].name;
};

document.getElementById("uploadBtnB").onchange = function () {
    document.getElementById("uploadFileB").value = this.files[0].name;
};

