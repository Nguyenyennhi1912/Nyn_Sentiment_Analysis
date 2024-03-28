mkdir -p ~/.streamlit/
echo "\
[general]\n\
email = \"youremail@domain\"\n\
" > ~/.streamlit/credentials.toml
echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableXsrfProtection=false\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml


