variable "project_name" {
  default = "depressive-sentiment-classifier"
}

variable "folders" {
  type    = list(string)
  default = ["bronze", "silver", "gold", "miscellaneous"]
}