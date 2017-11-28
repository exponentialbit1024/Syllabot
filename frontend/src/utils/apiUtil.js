import axios from 'axios';

export {getAns};

function getAns(searchTerm) {
  const url = `http://syllabot.herokuapp.com/textIn`;
  return axios.post(url, {
    input: searchTerm
}).then(response => response.data);
}
