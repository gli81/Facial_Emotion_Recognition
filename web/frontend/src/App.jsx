import { useState } from 'react';
import reactLogo from './assets/react.svg';
import viteLogo from '/vite.svg';
import './App.css';

function App() {
  // want a image submission form

  // image state
  // const [im, setIm] = useState(null)

  // image preview state
  const [imagePreview, setImagePreview] = useState(null);

  // image upload event handler
  const handleImageUpload = (e) => {
    // get the image file
    const file = e.target.files[0];
    // create a preview for the image
    const reader = new FileReader();
    reader.onloadend = () => {
      setImagePreview(reader.result)
      // console.log(reader.result)
    }
    reader.readAsDataURL(file)
  }

  // image submit event handler
  const handleImageSubmit = async () => {
    const ops = {
      method: 'POST',
      body: {
        image: imagePreview,
        model: "full"
      }
    }

    // send the image to the server
    try {
      console.log(ops)
      // const response = await fetch('http://localhost:5001/upload', {
      //   method: 'POST',
      //   body: formData,
      // })

      // const data = await response.json()
      // console.log(data)
    } catch (error) {
      console.error(error)
    }
  }

  return (
    <div className="App">
      <header className="App-header">
        <img src={reactLogo} className="App-logo" alt="react logo" />
        <img src={viteLogo} className="App-logo" alt="vite logo" />
        <h1>Upload an Image</h1>
        <input type="file" accept="image/*" onChange={handleImageUpload} />
        {imagePreview && (
          <img
            src={imagePreview}
            alt="image preview"
            style={{ width: '300px', height: '300px', objectFit: 'cover' }}
          />
        )}
        <button onClick={handleImageSubmit}>Submit</button>
      </header>
    </div>
  )
}

export default App
