package routes

/**
 * Route for uploading images
 * @api {post} /upload Upload an image
 */

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/gin-gonic/gin"
)

func UploadImage(c *gin.Context) {
	var body struct {
		Image string `json:"image" binding:"required"`
		Model string `json:"model" binding:"required"`
	}

	err := c.BindJSON(&body)

	// send request to the model
	url := "http://localhost:5000/inference"
	payload := map[string]string{
		"image":      body.Image,
		"model_name": body.Model,
	}
	fmt.Println(payload["model_name"])
	if err != nil {
		c.JSON(400, gin.H{
			"msg": err.Error(),
		})
		return
	}
	jsonData, err := json.Marshal(payload)
	if err != nil {
		c.JSON(500, gin.H{
			"msg": "Error building request",
		})
		return
	}
	type Data struct {
		Prediction int `json:"prediction"`
	}
	type Response struct {
		Msg  string `json:"msg"`
		Data Data   `json:"data"`
	}
	resp, err := http.Post(
		url,
		"application/json",
		bytes.NewBuffer(jsonData),
	)
	if err != nil {
		c.JSON(500, gin.H{
			"msg": "Error sending request",
		})
		return
	}
	defer resp.Body.Close() // don't forget!!
	// get response data
	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		c.JSON(500, gin.H{
			"msg": "Error reading response",
		})
		return
	}
	var result Response
	err = json.Unmarshal(respBody, &result)
	if err != nil {
		c.JSON(500, gin.H{
			"msg": "Error parsing response",
		})
		return
	}

	c.JSON(200, gin.H{
		"msg": "Suceess",
		"data": gin.H{
			"result": fmt.Sprintf(
				"%d", result.Data.Prediction,
			),
		},
	})
}
