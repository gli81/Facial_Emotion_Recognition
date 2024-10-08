package main

/**
 * Web interface for the Facial Emotion Recognition
 * handles user auth {TODO} and
 * pass image and model selection to the classifier
 */

import (
	// "fmt"
	// "os"

	"log"

	"github.com/gin-gonic/gin"
	"github.com/joho/godotenv"

	// "gorm.io/driver/postgres"
	// "gorm.io/gorm"
	"github.com/gin-contrib/cors"
	"github.com/gli81/Facial_Emotion_Recognition/web/interface/routes"
)

func init() {
	var err error
	err = godotenv.Load()
	if err != nil {
		log.Fatal("Error loading .env file")
	}
	// read dotenv for DB code commented out
	// 	host := os.Getenv("DB_HOST")
	// 	port := os.Getenv("DB_PORT")
	// 	user := os.Getenv("DB_USERNAME")
	// 	pw := os.Getenv("DB_PASSWORD")
	// 	if host == "" || port == "" || user == "" || pw == "" {
	// 		log.Fatal("Failed to load DB credentials")
	// 	}
	// 	dsn := fmt.Sprintf(
	// 		"host=%s port=%s user=%s password=%s dbname=postgres sslmode=disable",
	// 		host, port, user, pw,
	// 	)
	// 	_, err = gorm.Open(postgres.Open(dsn), &gorm.Config{})
	// 	if err != nil {
	// 		log.Fatal("Failed to connect to DB")
	// 	}
}

func main() {
	r := gin.Default()
	r.Use(cors.Default())
	// register routes
	r.POST("/upload", routes.UploadImage)
	r.Run(":5001")
}
