import cherrypy

class MatchServer(object):

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def index(self):
        return { 'success': True }

    @cherrypy.expose
    def ws(self):
        pass
