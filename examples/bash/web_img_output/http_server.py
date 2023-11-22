from http.server import SimpleHTTPRequestHandler
import socketserver

class CORSRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'X-Requested-With, Content-Type')
        super().end_headers()

if __name__ == '__main__':
    PORT = 8000
    with socketserver.TCPServer(('', PORT), CORSRequestHandler) as httpd:
        print(f'Serving at http://localhost:{PORT}')
        httpd.serve_forever()
