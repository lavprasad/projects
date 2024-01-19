const nav = document.querySelector('.navbar')
fetch('/navbar.html')
.then(res=>res.text())
.then(data=>{
    nav.innerHTML=data
})

const fott = document.querySelector('.foot')
fetch('/footer.html')
.then(res=>res.text())
.then(data=>{
    fott.innerHTML=data
})