const canvas = document.getElementById('canvas');
const resizer = document.createElement('canvas');
const increaseBtn = document.getElementById('increase');
const decreaseBtn = document.getElementById('decrease');
const sizeEL = document.getElementById('size');
const colorEl = document.getElementById('color');
const clearEl = document.getElementById('clear');
const sendBtn = document.getElementById('send')
const numberEL = document.getElementById('number');
const imageEL = document.getElementById('image')
const plotEL = document.getElementById('plot')

const ctx = canvas.getContext('2d');
const resizerCtx = resizer.getContext('2d');

let size = 10
let isPressed = false
colorEl.value = '#ffffff'
let color = '#ffffff'
let x
let y

canvas.addEventListener('mousedown', (e) => {
    isPressed = true

    x = e.offsetX
    y = e.offsetY
})

document.addEventListener('mouseup', (e) => {
    isPressed = false

    x = undefined
    y = undefined
})

canvas.addEventListener('mousemove', (e) => {
    if(isPressed) {
        const x2 = e.offsetX
        const y2 = e.offsetY

        drawCircle(x2, y2)
        drawLine(x, y, x2, y2)

        x = x2
        y = y2
    }
})

function drawCircle(x, y) {
    ctx.beginPath();
    ctx.arc(x, y, size, 0, Math.PI * 2)
    ctx.fillStyle = color
    ctx.fill()
}

function drawLine(x1, y1, x2, y2) {
    ctx.beginPath()
    ctx.moveTo(x1, y1)
    ctx.lineTo(x2, y2)
    ctx.strokeStyle = color
    ctx.lineWidth = size * 2
    ctx.stroke()
}

function updateSizeOnScreen() {
    sizeEL.innerText = size
}

increaseBtn.addEventListener('click', () => {
    size += 5

    if(size > 50) {
        size = 50
    }

    updateSizeOnScreen()
})

decreaseBtn.addEventListener('click', () => {
    size -= 5

    if(size < 5) {
        size = 5
    }

    updateSizeOnScreen()
})

colorEl.addEventListener('change', (e) => color = e.target.value)

clearEl.addEventListener('click', () => ctx.clearRect(0,0, canvas.width, canvas.height))

sendBtn.addEventListener('click', async (e) => {
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

    if (!imageData) {
        console.error('image data is not available.');
        return;
    }

    const imageBase64 = canvas.toDataURL().split(';base64,')[1];

    fetch('http://localhost:8000/image', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            base64: imageBase64,
        }),
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        console.log(data)
        if (data.number === 10) data.number = NaN
        numberEL.innerText = data.number
        imageEL.setAttribute("src", `../data/tests/image${data.image_id}.png`)
        plotEL.setAttribute("src", `../data/tests/plot${data.image_id}.png`)
    })
    .catch(error => {
        console.error('Error:', error);
    });
})
