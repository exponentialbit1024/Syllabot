import axios from 'axios';

export {getAns};

function getAns(searchTerm) {
  const url = `https://syllabot.herokuapp.com/textIn`;
  return axios.post(url, {
    input: searchTerm
}).then(response => response.data);
}
