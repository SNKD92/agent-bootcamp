provider "google" {
  project = "PROJECT_ID"
  region  = "us-central1"
  zone    = "us-central1-a"
}

resource "google_compute_instance" "example_vm" {
  name         = "example-vm"
  machine_type = "e2-medium"
  zone         = "us-central1-a"

  boot_disk {
    initialize_params {
      image = "debian-cloud/debian-11"
    }
  }

  network_interface {
    network = "default"

    # Optional external IP
    access_config {}
  }

  metadata = {
    ssh-keys = "youruser:${file("~/.ssh/id_rsa.pub")}"
  }
}
