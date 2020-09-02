mkdir -p ~/.streamlit/
echo "\
[general]\n\
email = \"nhorton04@gmail.com\"\n\
" > ~/.streamlit/credentials.toml
echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
