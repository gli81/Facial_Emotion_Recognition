import { useState } from 'react';
import reactLogo from './assets/react.svg';
import viteLogo from '/vite.svg';
import './App.css';

function App() {
  // image preview state
  const [imagePreview, setImagePreview] = useState(null);
  // image upload event
  const handleImageUpload = (e) => {
    // get the image file
    const file = e.target.files[0];
    // create a preview for the image
    const reader = new FileReader();
    reader.onloadend = () => {
      setImagePreview(reader.result);
      // console.log(reader.result)
    }
    reader.readAsDataURL(file);

    // tried to do resize in frontend, but python just easier to work with images
    // let files = file.files;
    // if (files.length === 0) {
    //   return;
    // }
    // let file2 = files[0];
    // fileReader = new FileReader();
    // fileReader.onload = function (e) {

    // }
  }

  // image submit event handler
  const handleImageSubmit = async () => {
    try {
      // const base64img = await resizeAndConvertToBase64(imagePreview)
      const ops = {
        image: imagePreview,
        model: "full"
      }

    // send the image to the server
      console.log(ops)
      const response = await fetch('http://localhost:5001/upload', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(ops),
      })

      const data = await response.json()
      console.log(data)
    } catch (error) {
      console.error(error)
    }
  }

  return (
    <div className="App">
      <header className="App-header">
        <h1>Upload an Image for Emotion Classification</h1>
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
