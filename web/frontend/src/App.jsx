/**
 * Frontend code to handle image upload and model selection.
 */

import { useState } from 'react';
import './App.css';

function App() {
  // image preview state
  const [imagePreview, setImagePreview] = useState(null);
  // radio button change
  const [model, setModel] = useState("full");
  const handleChange = (e) => {
    setModel(e.target.value);
  }
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
        model: model
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
        <br />
        {/* radio button to select model , choices are: full, left, right, upper, lower*/}
        <br />
        <input type="radio" id="full" name="model"
                value="full" checked={model === 'full'}
                onChange={handleChange} />FULL<br />
        <input type="radio" id="left" name="model"
                value="left" checked={model === 'left'}
                onChange={handleChange} />LEFT<br />
        <input type="radio" id="right" name="model"
                value="right" checked={model === 'right'}
                onChange={handleChange} />RIGHT<br />
        <input type="radio" id="upper" name="model"
                value="upper" checked={model === 'upper'}
                onChange={handleChange} />UPPER<br />
        <input type="radio" id="lower" name="model"
                value="lower" checked={model === 'lower'}
                onChange={handleChange} />LOWER<br />
        <button onClick={handleImageSubmit}>Submit</button>
      </header>
    </div>
  )
}

export default App
