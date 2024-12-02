# Audio Transcription Web App

This project is a web application for recording and uploading audio files, which are then transcribed using a server-side model. The application is built using Flask for the backend and HTML/JavaScript for the frontend.
This has no password protection, so make sure it's maintained and used purely in a trusted network

## Prerequisites

- Python 3.x
- Virtual environment support (venv)
- Ollama (for serving the transcription model)

## Setup Instructions

## 1. Clone the Repository

```bash
git clone https://github.com/yourusername/audiotranscription.git
cd audiotranscription
```

## 2. Set Up a Virtual Environment

Create and activate a virtual environment to manage dependencies.
further details on venv [here](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/)

#### On Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### On macOS/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install Dependencies

Use pip to install the required Python packages.

```bash
pip install -r requirements.txt
```

## 4. Configure Flask

Set up Flask environment variables and run the application.

```python app.py```
(make sure you modify this if you want to use this in production)

### Setting Up a Self-Signed SSL Certificate
This is necessary to get microphone access on most browsers, by allowing you to access the server via https

#### Prerequisites
#### MacOS
OpenSSL installed on your system:
macOS: Pre-installed or use Homebrew:
```brew install openssl```

#### Windows
Download from [link](https://slproweb.com/products/Win32OpenSSL.html)
Steps to Create a Self-Signed Certificate
1. Generate a Certificate and Key
Run the following OpenSSL command to generate a certificate (cert.pem) and private key (key.pem):

```openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes```

Command Explanation:
- x509: Create an X.509 certificate.
- newkey rsa:4096: Generate a new RSA key with a size of 4096 bits.
- keyout key.pem: Save the private key to key.pem.
- out cert.pem: Save the certificate to cert.pem.
- days 365: Certificate validity period (1 year).
- nodes: Skip encrypting the private key with a passphrase.

##### Fill Out Certificate Details
You will be prompted to enter details like:

- Country Name (e.g., AU)
- State or Province (e.g., Victoria)
- Locality Name (e.g., Melbourne)
- Organization Name
- Common Name (e.g., localhost for local testing)
- Set the Common Name to localhost if this certificate is for local development.


#### Place cert.pem and key.pem in your project directory.

#### Ignore the browser’s security warning (self-signed certificates are not trusted by default).

## 6. Download and Serve Ollama

Ollama is used to serve the transcription model. Follow these steps to set it up:

#### Download Ollama

Visit [Ollama's website](https://ollama.com) to download and install the software.

#### Serve the Model

Start the Ollama server and specify the port number.
!!!Modify line 32 of app.py to your computer's ollama port number (likely 11434)

##### On Windows
Modify your environmental variables set
```OLLAMA_HOST = 0.0.0.0:<PORTNUMBER>```

skip this step if you don't want to customise your port
[full guide here](https://github.com/ollama/ollama/blob/main/docs/faq.md#how-do-i-configure-ollama-server)

```
ollama serve
```

##### On macOS
[guide to port changing here](https://github.com/ollama/ollama/blob/main/docs/faq.md#how-do-i-configure-ollama-server)
```
ollama serve
```

## 7. VPN Setup (Optional)

If you need to access the server from a different network, you can use a VPN service like ZeroTier or Tailscale.

#### ZeroTier

1. Sign up and install ZeroTier from [ZeroTier's website](https://www.zerotier.com/).
2. Create a network and join it using the ZeroTier client.
3. Authorize your device in the ZeroTier Central web interface.

#### Tailscale

1. Sign up and install Tailscale from [Tailscale's website](https://tailscale.com/).
2. Log in and connect your device to your Tailscale network.
3. Use the Tailscale IP address to access the server.

## Usage

1. Open your web browser and navigate to `https://localhost:5000`. *Don't forget the https, most browsers limit microphone access on http*
2. Use the interface to record or upload audio files.
3. The transcription will be displayed after processing.

## Troubleshooting

- Ensure that your microphone permissions are enabled in your browser.
- Check that the Ollama server is running and accessible.
- Verify that Flask is running on the correct port.

## Contributing

Feel free to submit issues or pull requests to improve the project.


